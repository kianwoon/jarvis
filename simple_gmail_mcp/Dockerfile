FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY index.js ./

ENTRYPOINT ["node", "index.js"]