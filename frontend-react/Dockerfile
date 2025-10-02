# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies needed for node-gyp
RUN apk add --no-cache python3 make g++

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies for building)
RUN npm ci --include=dev

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Set working directory
WORKDIR /app

# Install serve to serve static files
RUN npm install -g serve

# Copy built application from builder stage
COPY --from=builder /app/build ./build

# Copy package.json for metadata
COPY package*.json ./

# Create a simple health check endpoint
RUN echo '#!/bin/sh\necho "OK"' > /app/health.sh && chmod +x /app/health.sh

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S reactuser -u 1001

# Change ownership of the app directory
RUN chown -R reactuser:nodejs /app

# Switch to non-root user
USER reactuser

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

# Start the application using serve
CMD ["serve", "-s", "build", "-l", "3000"]
