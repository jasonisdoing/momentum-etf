import { createFastApiProxy } from "@/lib/fastapi-proxy";

export const dynamic = "force-dynamic";

const proxy = createFastApiProxy({
  PUT: {
    path: "/internal/strategy-mix/settings",
    error: "설정을 저장하지 못했습니다.",
    forwardBody: true,
  },
});

export const PUT = proxy.PUT!;
