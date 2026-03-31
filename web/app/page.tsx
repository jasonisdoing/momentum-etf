import { PageFrame } from "./components/PageFrame";

export const dynamic = "force-dynamic";

export default function HomePage() {
  return (
    <PageFrame title="Home">
      <div className="card">
        <div className="card-body py-5 text-center">
          <div className="empty">
            <p className="empty-title">Home</p>
            <p className="empty-subtitle text-secondary">
              준비 중입니다. 왼쪽 메뉴에서 대시보드를 선택해 주세요.
            </p>
          </div>
        </div>
      </div>
    </PageFrame>
  );
}

