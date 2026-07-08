import { PageFrame } from "../components/PageFrame";

export const dynamic = "force-dynamic";

export default function TopPickPage() {
  return (
    <PageFrame title="탑픽 비중">
      <div className="appPageStack">
        <div className="card appCard">
          <div className="card-body">
            <h2 style={{ fontSize: "1.05rem", fontWeight: 800, marginBottom: 8 }}>비중</h2>
            <p style={{ color: "#64748b", margin: 0 }}>
              등록된 탑픽 포트폴리오의 현재 비중과 목표 비중을 조회하는 화면입니다.
            </p>
          </div>
        </div>
      </div>
    </PageFrame>
  );
}
