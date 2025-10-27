# Nginx에서 Open Graph 메타 태그 추가하기

## 1. Nginx 설정 파일 수정

Ubuntu 서버에서 Nginx 설정 파일을 수정합니다:

```bash
sudo nano /etc/nginx/sites-available/etf.dojason.com
```

## 2. 메타 태그 주입 설정 추가

기존 `location /` 블록에 다음 내용을 추가:

```nginx
server {
    listen 80;
    server_name etf.dojason.com;

    # Streamlit 앱으로 프록시
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # HTML 응답에 메타 태그 주입
        sub_filter '</head>' '
    <meta property="og:title" content="Momentum ETF - 모멘텀 투자 전략 대시보드" />
    <meta property="og:description" content="데이터 기반 모멘텀 투자 전략으로 포트폴리오를 관리하세요. 실시간 추천 및 성과 분석을 제공합니다." />
    <meta property="og:image" content="https://etf.dojason.com/static/og-image.png" />
    <meta property="og:url" content="https://etf.dojason.com/" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="Momentum ETF" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="Momentum ETF - 모멘텀 투자 전략 대시보드" />
    <meta name="twitter:description" content="데이터 기반 모멘텀 투자 전략으로 포트폴리오를 관리하세요." />
    <meta name="twitter:image" content="https://etf.dojason.com/static/og-image.png" />
    <meta name="description" content="데이터 기반 모멘텀 투자 전략으로 포트폴리오를 관리하세요. 실시간 추천 및 성과 분석을 제공합니다." />
</head>';
        sub_filter_once on;
        sub_filter_types text/html;
    }
    
    # 정적 파일 제공 (OG 이미지 등)
    location /static/ {
        alias /path/to/momentum-etf/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

## 3. Nginx 설정 테스트 및 재시작

```bash
# 설정 파일 문법 체크
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

## 4. OG 이미지 생성

`static/og-image.png` 파일을 생성해야 합니다. 권장 사이즈:
- **1200 x 630 픽셀** (Facebook, LinkedIn)
- **1200 x 675 픽셀** (Twitter)

이미지에는 다음 내용을 포함하는 것이 좋습니다:
- 서비스 이름: "Momentum ETF"
- 간단한 설명: "모멘텀 투자 전략 대시보드"
- 브랜드 컬러 (#D94D2B)
- 차트나 그래프 아이콘

## 5. 테스트

메타 태그가 제대로 적용되었는지 확인:

```bash
curl -I https://etf.dojason.com/
```

또는 다음 도구들을 사용:
- Facebook Sharing Debugger: https://developers.facebook.com/tools/debug/
- Twitter Card Validator: https://cards-dev.twitter.com/validator
- LinkedIn Post Inspector: https://www.linkedin.com/post-inspector/

## 주의사항

1. **캐시 문제**: 메신저 앱들은 링크 미리보기를 캐시합니다. 변경 후 즉시 반영되지 않을 수 있습니다.
2. **HTTPS**: 일부 플랫폼은 HTTPS를 요구합니다.
3. **이미지 크기**: OG 이미지는 8MB 이하여야 합니다.
4. **절대 URL**: 이미지 URL은 반드시 절대 경로여야 합니다.
