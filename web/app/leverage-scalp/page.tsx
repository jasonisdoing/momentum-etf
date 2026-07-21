import { fetchFastApiJson } from "../../lib/internal-api";
import { LeverageScalpClient } from "./LeverageScalpClient";

export const dynamic = "force-dynamic";

type SystemConfig = { ma_type: string };

export default async function LeverageScalpPage() {
  // 선물 참고 이동평균선 오버레이 종류(SMA/EMA)를 config.py 에서 받아 화면에 전달(표시·계산 전용).
  const cfg = await fetchFastApiJson<SystemConfig>("/internal/system/config");
  return <LeverageScalpClient maType={cfg.ma_type} />;
}
