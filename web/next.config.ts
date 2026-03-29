import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  turbopack: {
    // 루트 추론 경고를 막기 위해 web 디렉터리를 명시합니다.
    root: path.join(__dirname),
  },
};

export default nextConfig;
