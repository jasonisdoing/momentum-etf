import { PageFrame } from "../components/PageFrame";

export const dynamic = "force-dynamic";

export default function TopPickSettingsPage() {
  return (
    <PageFrame title="탑픽 설정">
      <div className="appPageStack">
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 8 }}>설정</h2>
            <p style={{ color: "#64748b", margin: 0 }}>
              탑픽 포트폴리오 등록, 목표 비중 변경, 백테스트 설정을 관리하는 화면입니다.
            </p>
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
