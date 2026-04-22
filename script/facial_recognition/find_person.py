import os

# Set logging env vars before importing onnxruntime/insightface to reduce noisy C++ warnings.
for _k, _v in (
    ("ORT_LOG_SEVERITY_LEVEL", "3"),
    ("ONNXRUNTIME_LOG_SEVERITY_LEVEL", "3"),
):
    os.environ.setdefault(_k, _v)

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Cache directory (not a single fixed file).
# Must be writable both in container and local environments; /tmp avoids path/permission issues.
CACHE_DIR = os.path.join("/tmp", "openclaw_target_person_cache")


def iter_images(folder: str):
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn.lower())[1]
            if ext in IMG_EXTS:
                yield os.path.join(root, fn)


def read_image(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def pick_largest_face(faces):
    best = None
    best_area = -1.0
    for f in faces:
        x1, y1, x2, y2 = map(float, f.bbox)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best = f
    return best


def l2_normalize(v: np.ndarray, eps=1e-12):
    n = np.linalg.norm(v)
    return v / max(n, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json_output(payload: dict, out_path: Optional[str]):
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(text, encoding="utf-8")


def list_target_person_ids(target_dir: str) -> List[str]:
    """
    新模式（强制）:
      target_dir/
        <person_id_1>/*.jpg
        <person_id_2>/*.jpg
    """
    target_dir = os.path.abspath(target_dir)
    ids = []
    for name in sorted(os.listdir(target_dir)):
        if name.startswith("."):
            continue
        full = os.path.join(target_dir, name)
        if os.path.isdir(full):
            ids.append(name)
    return ids


def get_person_images(target_dir: str, person_id: str) -> List[str]:
    person_dir = os.path.join(target_dir, person_id)
    return sorted(iter_images(person_dir))


def build_target_fingerprint(target_dir: str, model_name: str, det_size: Tuple[int, int]):
    """
    生成目标目录指纹（多目标模式）
    """
    target_dir = os.path.abspath(target_dir)
    person_ids = list_target_person_ids(target_dir)

    h = hashlib.sha256()
    h.update(target_dir.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    h.update(str(det_size).encode("utf-8"))

    all_entries: List[Tuple[str, str]] = []
    for pid in person_ids:
        files = get_person_images(target_dir, pid)
        for p in files:
            st = os.stat(p)
            rel = os.path.relpath(p, target_dir)
            row = f"{pid}|{rel}|{st.st_size}|{int(st.st_mtime_ns)}"
            h.update(row.encode("utf-8"))
            all_entries.append((pid, p))

    return h.hexdigest(), all_entries, person_ids


def get_cache_file_path(target_dir: str, model_name: str, det_size: Tuple[int, int]):
    raw = f"{os.path.abspath(target_dir)}|{model_name}|{det_size}|multi-target-v1"
    key = hashlib.md5(raw.encode("utf-8")).hexdigest()
    ensure_dir(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{key}.npz")


def save_target_cache(
    cache_path: str,
    target_db: List[Dict[str, object]],
    stats: dict,
    fingerprint: str,
    target_dir: str,
    model_name: str,
    det_size: Tuple[int, int],
):
    person_ids = [str(x["person_id"]) for x in target_db]
    emb_matrix = np.stack([x["emb"] for x in target_db], axis=0).astype(np.float32)
    np.savez(
        cache_path,
        person_ids=np.array(person_ids),
        emb_matrix=emb_matrix,
        stats_json=np.array([json.dumps(stats, ensure_ascii=False)]),
        fingerprint=np.array([fingerprint]),
        target_dir=np.array([os.path.abspath(target_dir)]),
        model_name=np.array([model_name]),
        det_w=np.array([det_size[0]], dtype=np.int32),
        det_h=np.array([det_size[1]], dtype=np.int32),
    )


def load_target_cache(
    cache_path: str,
    expected_fingerprint: str,
    target_dir: str,
    model_name: str,
    det_size: Tuple[int, int],
):
    if not os.path.isfile(cache_path):
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        person_ids = [str(x) for x in data["person_ids"]]
        emb_matrix = data["emb_matrix"].astype(np.float32)
        stats_json = str(data["stats_json"][0])

        cached_fingerprint = str(data["fingerprint"][0])
        cached_target_dir = str(data["target_dir"][0])
        cached_model_name = str(data["model_name"][0])
        cached_det_size = (int(data["det_w"][0]), int(data["det_h"][0]))

        if cached_fingerprint != expected_fingerprint:
            return None
        if os.path.abspath(target_dir) != cached_target_dir:
            return None
        if model_name != cached_model_name:
            return None
        if det_size != cached_det_size:
            return None
        if emb_matrix.ndim != 2 or emb_matrix.shape[0] != len(person_ids):
            return None

        target_db: List[Dict[str, object]] = []
        for i, person_id in enumerate(person_ids):
            target_db.append({"person_id": person_id, "emb": l2_normalize(emb_matrix[i])})

        stats = json.loads(stats_json)
        return target_db, stats
    except Exception:
        return None


def build_target_db(app: FaceAnalysis, target_dir: str):
    target_dir = os.path.abspath(target_dir)
    person_ids = list_target_person_ids(target_dir)
    if not person_ids:
        raise RuntimeError(
            "No person folders found in target_dir. Use new layout: target/<person_id>/*.jpg"
        )

    target_db: List[Dict[str, object]] = []
    people_stats: List[dict] = []
    total_files = 0

    for pid in person_ids:
        embs = []
        details = []
        files = get_person_images(target_dir, pid)
        total_files += len(files)

        for p in files:
            img = read_image(p)
            if img is None:
                details.append(
                    {"path": p, "status": "skipped", "reason": "image_read_failed"}
                )
                continue

            faces = app.get(img)
            if not faces:
                details.append(
                    {"path": p, "status": "skipped", "reason": "no_face_detected"}
                )
                continue

            f = pick_largest_face(faces)
            emb = l2_normalize(f.embedding.astype(np.float32))
            embs.append(emb)
            details.append({"path": p, "status": "used", "faces_detected": len(faces)})

        person_stat = {
            "person_id": pid,
            "files_total": len(files),
            "used": len(embs),
            "skipped": max(0, len(files) - len(embs)),
            "details": details,
        }
        people_stats.append(person_stat)

        if len(embs) == 0:
            continue
        mean_emb = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0))
        target_db.append({"person_id": pid, "emb": mean_emb})

    if not target_db:
        raise RuntimeError(
            f"No usable faces found in target_dir={target_dir}. Ensure each person folder has at least one clear face."
        )

    stats = {
        "mode": "multi_target",
        "persons_total": len(person_ids),
        "persons_used": len(target_db),
        "persons_skipped": len(person_ids) - len(target_db),
        "target_files_count": total_files,
        "persons": people_stats,
    }
    return target_db, stats


def get_target_db(app: FaceAnalysis, target_dir: str, model_name: str, det_size: Tuple[int, int]):
    fingerprint, all_entries, person_ids = build_target_fingerprint(target_dir, model_name, det_size)
    cache_path = get_cache_file_path(target_dir, model_name, det_size)

    cached = load_target_cache(
        cache_path=cache_path,
        expected_fingerprint=fingerprint,
        target_dir=target_dir,
        model_name=model_name,
        det_size=det_size,
    )
    if cached is not None:
        target_db, stats = cached
        return target_db, stats, True, cache_path, len(all_entries), len(person_ids), fingerprint

    target_db, stats = build_target_db(app, target_dir)
    save_target_cache(
        cache_path=cache_path,
        target_db=target_db,
        stats=stats,
        fingerprint=fingerprint,
        target_dir=target_dir,
        model_name=model_name,
        det_size=det_size,
    )
    return target_db, stats, False, cache_path, len(all_entries), len(person_ids), fingerprint


def best_target_for_embedding(emb: np.ndarray, target_db: List[Dict[str, object]]):
    best_person_id = None
    best_score = -1.0
    for item in target_db:
        person_id = str(item["person_id"])
        target_emb = item["emb"]
        s = cosine(emb, target_emb)
        if s > best_score:
            best_score = s
            best_person_id = person_id
    return best_person_id, float(best_score)


def score_photo(app: FaceAnalysis, img_path: str, target_db: List[Dict[str, object]]):
    img = read_image(img_path)
    if img is None:
        return {
            "path": img_path,
            "max_score": None,
            "best_person_id": None,
            "num_faces": 0,
            "reason": "image_read_failed",
        }

    faces = app.get(img)
    if not faces:
        return {
            "path": img_path,
            "max_score": None,
            "best_person_id": None,
            "num_faces": 0,
            "reason": "no_face_detected",
        }

    max_score = -1.0
    best_person_id = None
    for f in faces:
        emb = l2_normalize(f.embedding.astype(np.float32))
        person_id, score = best_target_for_embedding(emb, target_db)
        if score > max_score:
            max_score = score
            best_person_id = person_id

    return {
        "path": img_path,
        "max_score": float(max_score),
        "best_person_id": best_person_id,
        "num_faces": len(faces),
        "reason": "ok",
    }


def verify_one_image(app: FaceAnalysis, img_path: str, target_db: List[Dict[str, object]], threshold: float):
    img = read_image(img_path)
    if img is None:
        return {
            "path": img_path,
            "matched": False,
            "score": None,
            "best_person_id": None,
            "matched_person_ids": [],
            "num_faces": 0,
            "face_matches": [],
            "reason": "image_read_failed",
        }

    faces = app.get(img)
    if not faces:
        return {
            "path": img_path,
            "matched": False,
            "score": None,
            "best_person_id": None,
            "matched_person_ids": [],
            "num_faces": 0,
            "face_matches": [],
            "reason": "no_face_detected",
        }

    best_score = -1.0
    best_person_id = None
    matched_ids = set()
    face_matches = []

    for i, f in enumerate(faces):
        emb = l2_normalize(f.embedding.astype(np.float32))
        person_id, score = best_target_for_embedding(emb, target_db)
        matched = score >= threshold
        if matched and person_id:
            matched_ids.add(person_id)
        if score > best_score:
            best_score = score
            best_person_id = person_id
        face_matches.append(
            {
                "face_index": i,
                "person_id": person_id,
                "score": float(score),
                "matched": matched,
                "bbox": [float(x) for x in f.bbox],
            }
        )

    return {
        "path": img_path,
        "matched": best_score >= threshold,
        "score": float(best_score),
        "best_person_id": best_person_id,
        "matched_person_ids": sorted(matched_ids),
        "num_faces": len(faces),
        "face_matches": face_matches,
        "reason": "ok",
    }


def run_search(
    app: FaceAnalysis,
    target_dir: str,
    photos_dir: str,
    threshold: float,
    topk: int,
    out_path: Optional[str],
    model_name: str,
    det_size: Tuple[int, int],
):
    target_db, stats, from_cache, cache_path, target_files_count, target_person_count, fingerprint = (
        get_target_db(app, target_dir, model_name, det_size)
    )

    results = []
    checked = 0

    for p in iter_images(photos_dir):
        checked += 1
        r = score_photo(app, p, target_db)
        if r["max_score"] is None:
            continue
        if r["max_score"] >= threshold:
            results.append(r)

    results.sort(key=lambda x: x["max_score"], reverse=True)
    hits = results[:topk]

    payload = {
        "mode": "search",
        "target_mode": "multi_target",
        "threshold": threshold,
        "target_dir": os.path.abspath(target_dir),
        "target_files_count": target_files_count,
        "target_person_count": target_person_count,
        "cache_path": cache_path,
        "cache_used": from_cache,
        "target_fingerprint": fingerprint,
        "target_stats": stats,
        "photos_dir": os.path.abspath(photos_dir),
        "photos_checked": checked,
        "hits_count": len(results),
        "hits_topk": len(hits),
        "hits": hits,
    }

    write_json_output(payload, out_path)


def run_verify(
    app: FaceAnalysis,
    target_dir: str,
    image_path: str,
    threshold: float,
    out_path: Optional[str],
    model_name: str,
    det_size: Tuple[int, int],
):
    target_db, stats, from_cache, cache_path, target_files_count, target_person_count, fingerprint = (
        get_target_db(app, target_dir, model_name, det_size)
    )
    result = verify_one_image(app, image_path, target_db, threshold)

    payload = {
        "mode": "verify",
        "target_mode": "multi_target",
        "threshold": threshold,
        "target_dir": os.path.abspath(target_dir),
        "target_files_count": target_files_count,
        "target_person_count": target_person_count,
        "cache_path": cache_path,
        "cache_used": from_cache,
        "target_fingerprint": fingerprint,
        "target_stats": stats,
        "result": result,
    }

    write_json_output(payload, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="人脸搜索 / 单图验证（多目标模式：target/<person_id>/*.jpg，自动缓存 embedding）"
    )
    parser.add_argument("mode", choices=["search", "verify"], help="search=批量搜索, verify=单图验证")
    parser.add_argument(
        "target_path",
        help="目标人物目录（仅支持新模式）：target/<person_id>/*.jpg",
    )
    parser.add_argument("input_path", help="search 模式传照片文件夹路径；verify 模式传单张图片路径")
    parser.add_argument("--threshold", type=float, default=0.42, help="余弦相似度阈值")
    parser.add_argument("--topk", type=int, default=50, help="search 模式输出前 top-k 条命中")
    parser.add_argument("--out", type=str, default=None, help="输出 JSON 文件路径")
    parser.add_argument("--model", type=str, default="buffalo_l", help="InsightFace 模型名")
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("W", "H"),
        help="检测尺寸，例如 --det-size 640 640",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.target_path):
        raise ValueError(f"target_path 不是有效文件夹: {args.target_path}")

    if args.mode == "search" and not os.path.isdir(args.input_path):
        raise ValueError(f"search 模式下 input_path 必须是文件夹: {args.input_path}")

    if args.mode == "verify" and not os.path.isfile(args.input_path):
        raise ValueError(f"verify 模式下 input_path 必须是单张图片文件: {args.input_path}")

    det_size = (int(args.det_size[0]), int(args.det_size[1]))
    model_name = args.model

    app = FaceAnalysis(name=model_name)
    app.prepare(ctx_id=-1, det_size=det_size)

    if args.mode == "search":
        run_search(
            app=app,
            target_dir=args.target_path,
            photos_dir=args.input_path,
            threshold=args.threshold,
            topk=args.topk,
            out_path=args.out,
            model_name=model_name,
            det_size=det_size,
        )
    else:
        run_verify(
            app=app,
            target_dir=args.target_path,
            image_path=args.input_path,
            threshold=args.threshold,
            out_path=args.out,
            model_name=model_name,
            det_size=det_size,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[openclaw-facial] fatal: {e}", file=sys.stderr)
        sys.exit(1)