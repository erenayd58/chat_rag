# The console. Three stages, so neither the npm cache nor the dev dependencies
# reach the image that runs.
#
# Node 22 is the active LTS and is what Next.js 15 is tested against; the tag
# is pinned to a minor so a rebuild months from now produces the same runtime
# rather than whatever "22" has become.
FROM node:22.20-alpine AS deps

WORKDIR /app

# Only the manifests, so this layer is rebuilt when the dependencies change
# and not when a component does.
COPY package.json package-lock.json ./
# `npm ci` and not `npm install`: the lockfile is the declared source, and
# `ci` fails on a package.json it does not match instead of quietly resolving
# something newer.
RUN npm ci


FROM node:22.20-alpine AS builder

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# The build must not need to know where the application is. It used to: a
# `rewrites()` entry read CHAT_RAG_API_URL here and Next.js froze the resolved
# destination into .next/routes-manifest.json, so the image carried the
# developer's `127.0.0.1` whatever the container was told at run time. The
# forwarding is a route handler now (`lib/api/proxy.ts`), which reads the
# address per request -- so this stage takes no build argument at all, and one
# image runs against any backend.
RUN npm run build


FROM node:22.20-alpine AS runner

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    # Every interface: a container's port mapping only reaches a process
    # listening on all of them, which is the same reason the application binds
    # 0.0.0.0.
    HOSTNAME=0.0.0.0

# Runs as a normal user. The image's own user, renamed to match the
# application container's, so a file written by either is legible in the other.
RUN addgroup -g 10001 app && adduser -u 10001 -G app -S -H app

WORKDIR /app

# `output: 'standalone'` traces which of node_modules the server actually
# reaches and copies only that, next to a server.js that needs no `next` CLI.
# The static assets and the public directory are not traced -- they are served
# from disk -- so they are copied beside it.
COPY --from=builder --chown=app:app /app/.next/standalone ./
COPY --from=builder --chown=app:app /app/.next/static ./.next/static

USER app

EXPOSE 3000

# Cheap and honest: it asks this server for a page it renders itself, so it
# says whether the console is serving without claiming anything about the
# application behind it. The application has its own health check, and the
# compose file waits on that one before starting this container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD node -e "require('http').get({host:'127.0.0.1',port:3000,path:'/',timeout:4000},r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))"

CMD ["node", "server.js"]
