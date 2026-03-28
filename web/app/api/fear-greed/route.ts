import { NextResponse } from "next/server";

import { parseFearGreedSummary } from "@/lib/fear-greed";

export const dynamic = "force-dynamic";

const CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata";

export async function GET() {
  try {
    const response = await fetch(CNN_FEAR_GREED_URL, {
      cache: "no-store",
      headers: {
        Accept: "application/json,text/plain,*/*",
        Referer: "https://edition.cnn.com/",
        "User-Agent":
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
      },
    });

    if (!response.ok) {
      throw new Error(`CNN 공포탐욕지수 조회 실패: ${response.status}`);
    }

    const payload = (await response.json()) as unknown;
    const summary = parseFearGreedSummary(payload);
    return NextResponse.json(summary);
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "CNN 공포탐욕지수를 불러오지 못했습니다.",
      },
      { status: 503 },
    );
  }
}
