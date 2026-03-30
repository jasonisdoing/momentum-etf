import { Metadata } from "next";
import { PageFrame } from "../components/PageFrame";
import { AccountStocksManager } from "./AccountStocksManager";

export const metadata: Metadata = {
  title: "Momentum ETF - 포트폴리오(계좌별 비중)",
  description: "계좌별 목표 포트폴리오(종목 및 비율) 관리",
};

export default function AccountStocksPage() {
  return (
    <PageFrame title="계좌별 종목" fullHeight fullWidth>
      <AccountStocksManager />
    </PageFrame>
  );
}
