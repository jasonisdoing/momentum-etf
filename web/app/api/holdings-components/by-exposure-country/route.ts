import { NextRequest } from "next/server";
import { fetchFastApiJson } from "../../../../lib/internal-api";
import { jsonNoStore } from "../../../../lib/no-store-response";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const code = request.nextUrl.searchParams.get("exposure_country_code");
    if (!code) {
      return jsonNoStore(
        { error: "exposure_country_code 파라미터가 필요합니다." },
        { status: 400 },
      );
    }
    const data = await fetchFastApiJson(
      `/internal/holdings-components/by-exposure-country?exposure_country_code=${encodeURIComponent(code)}`,
    );
    return jsonNoStore(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "데이터를 불러오지 못했습니다.";
    return jsonNoStore({ error: message }, { status: 500 });
  }
}
