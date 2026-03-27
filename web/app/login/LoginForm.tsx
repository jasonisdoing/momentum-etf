"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";

function buildGoogleLoginHref(nextPath: string): string {
  const params = new URLSearchParams({ next: nextPath });
  return `/api/auth/google/start?${params.toString()}`;
}

export function LoginForm() {
  const searchParams = useSearchParams();
  const nextPath = searchParams.get("next") || "/dashboard";
  const error = searchParams.get("error");

  return (
    <div className="loginCard">
      <div className="loginHeader">
        <h1>Momentum ETF</h1>
      </div>

      {error ? <div className="bannerError">{error}</div> : null}

      <Link className="primaryButton loginButton loginGoogleButton" href={buildGoogleLoginHref(nextPath)}>
        Google로 로그인
      </Link>
    </div>
  );
}
