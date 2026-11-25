# Build context is the repository root so the UI can bundle the eval results.
FROM node:20-alpine AS build
WORKDIR /src/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
COPY eval/results /src/eval/results
COPY eval/thresholds.json /src/eval/thresholds.json
RUN npm run build

FROM nginx:1.27-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/frontend/dist /usr/share/nginx/html
EXPOSE 80
