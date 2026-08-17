import { NextResponse, type NextRequest } from "next/server";

import { getAuthCookieName, verifySessionToken } from "./lib/auth";

function isPublicPath(pathname: string): boolean {
  return pathname === "/login" || pathname === "/api/live" || pathname.startsWith("/api/auth/");
}

function isStaticPath(pathname: string): boolean {
  return (
    pathname.startsWith("/_next/") ||
    pathname.startsWith("/favicon") ||
    pathname.startsWith("/static/") ||
    pathname === "/robots.txt" ||
    // PWA 설치 파일 — 로그인 화면으로 돌려보내면 홈 화면 추가가 아이콘 없이 붙는다.
    // 앱 이름·색·아이콘뿐이라 가려야 할 내용이 없다.
    pathname === "/manifest.webmanifest" ||
    pathname === "/apple-touch-icon.png" ||
    pathname === "/sw.js" ||
    pathname.startsWith("/icon-")
  );
}

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  if (isStaticPath(pathname)) {
    return NextResponse.next();
  }

  const cookieValue = request.cookies.get(getAuthCookieName())?.value;
  const session = await verifySessionToken(cookieValue);

  if (pathname === "/login") {
    if (session) {
      return NextResponse.redirect(new URL("/dashboard", request.url));
    }
    return NextResponse.next();
  }

  if (isPublicPath(pathname)) {
    return NextResponse.next();
  }

  if (session) {
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "로그인이 필요합니다." }, { status: 401 });
  }

  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set("next", pathname);
  return NextResponse.redirect(loginUrl);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
