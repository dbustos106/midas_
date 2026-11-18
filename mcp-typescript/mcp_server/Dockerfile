FROM node:22-alpine
WORKDIR /app

RUN corepack enable

COPY package.json pnpm-lock.yaml* ./
RUN pnpm install

COPY . .

ENV PORT=3000
EXPOSE 3000

CMD ["pnpm", "exec", "tsx", "server.ts"]
