import Link from "next/link";

const appItems = [
  {
    href: "/dashboard",
    title: "대시보드",
    description: "핵심 지표와 계좌 요약을 보는 메인 화면",
    icon: "▣",
  },
  {
    href: "/cash",
    title: "자산관리",
    description: "계좌별 원금과 현금을 직접 관리",
    icon: "￦",
  },
  {
    href: "/snapshots",
    title: "스냅샷",
    description: "일별 스냅샷 기록과 계좌 상세 확인",
    icon: "◫",
  },
  {
    href: "/market",
    title: "ETF 마켓",
    description: "ETF 시장 현황과 필터 기반 조회",
    icon: "◈",
  },
  {
    href: "/py/rank",
    title: "Python 순위",
    description: "기존 Streamlit 분석 화면 바로가기",
    icon: "△",
  },
];

export default function HomePage() {
  return (
    <main className="shell">
      <section className="pageHeaderCompact">
        <div><h1>앱 메뉴</h1></div>
      </section>

      <section className="section sectionLauncher">
        <div className="launcherGrid">
          {appItems.map((item) => (
            <Link key={item.href} href={item.href} className="launcherCard">
              <div className="launcherIcon" aria-hidden="true">{item.icon}</div>
              <div className="launcherTitle">{item.title}</div>
              <div className="launcherDescription">{item.description}</div>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
