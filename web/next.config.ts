import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  experimental: {
    // 개발 디스크 캐시가 8GB까지 쌓여 시작 때 서버가 응답 없이 멈춘 적이 있어 끈다.
    turbopackFileSystemCacheForDev: false,
  },
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
