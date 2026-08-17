import Link from "next/link";

import styles from "./mobile.module.css";

export const dynamic = "force-dynamic";

/** 모바일 홈 — 화면 진입점만 둔다. 위쪽은 비워 두고 아래에서 손가락이 닿는 곳에 버튼을 놓는다. */
export default function MobileHomePage() {
  return (
    <div className={`${styles.page} ${styles.homePage}`}>
      <div className={styles.homeTop} />
      <div className={styles.homeMenu}>
        <Link href="/m/assets" className={styles.homeButton}>
          자산
        </Link>
      </div>
    </div>
  );
}
