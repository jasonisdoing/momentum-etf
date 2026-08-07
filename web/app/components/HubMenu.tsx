"use client";

import Link from "next/link";

import { NAV_GROUPS, ROOT_ITEMS, type NavIcon } from "@/lib/nav-menu";

// 홈(허브) 메뉴 레일 — 사이드바와 같은 메뉴 정의(nav-menu.ts)를 대시보드 우측에
// 지표 카드와 같은 카드 스타일 1장으로 통합해 보여준다("메뉴 겸 대시보드").
// P4에서 각 항목에 라이브 요약(레짐/알람/배치 상태 등)을 붙일 예정.

function ItemIcon({ icon }: { icon: NavIcon }) {
  if (typeof icon === "string") {
    return <span className="hubMenuEmoji">{icon}</span>;
  }
  const Icon = icon;
  return <Icon size={16} stroke={1.9} />;
}

export function HubMenu() {
  const sections = [
    { id: "root", title: "바로가기", items: ROOT_ITEMS },
    ...NAV_GROUPS.map((group) => ({ id: group.id, title: group.title, items: group.items })),
  ];

  return (
    <div className="card appCard hubMenuCard">
      <div className="card-body hubMenuCardBody">
        {sections.map((section) => (
          <div key={section.id} className="hubMenuSection">
            <div className="hubMenuGroupTitle">{section.title}</div>
            <div className="hubMenuItems">
              {section.items.map((item) => (
                <Link key={item.href} href={item.href} className="hubMenuItem">
                  <span className="hubMenuItemIcon">
                    <ItemIcon icon={item.icon} />
                  </span>
                  <span className="hubMenuItemLabel">{item.label}</span>
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>
      <style jsx global>{`
        .hubMenuCard {
          position: sticky;
          top: 72px;
        }
        .hubMenuCardBody {
          display: flex;
          flex-direction: column;
          gap: 12px;
          padding: 0.85rem 0.9rem;
        }
        .hubMenuGroupTitle {
          margin-bottom: 4px;
          font-size: var(--fs-sm);
          font-weight: 800;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: var(--text-muted, #8a94a6);
        }
        .hubMenuItems {
          display: flex;
          flex-direction: column;
          gap: 1px;
        }
        .hubMenuItem {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 4px 8px;
          border-radius: 7px;
          color: inherit;
          font-size: var(--fs-base);
          font-weight: 600;
          text-decoration: none;
        }
        .hubMenuItem:hover {
          background: rgba(99, 102, 241, 0.09);
          color: inherit;
        }
        .hubMenuItemIcon {
          display: inline-flex;
          width: 18px;
          justify-content: center;
          color: var(--text-muted, #64748b);
        }
        .hubMenuEmoji {
          font-size: var(--fs-base);
          line-height: 1;
        }
      `}</style>
    </div>
  );
}
