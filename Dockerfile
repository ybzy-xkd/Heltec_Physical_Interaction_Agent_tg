FROM node:latest
COPY . /videobot
WORKDIR /videobot
RUN npm install 
EXPOSE 3000
ENTRYPOINT ["npm", "run"]
CMD ["start"]