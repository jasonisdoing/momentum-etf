# 🚀 Open Graph 메타 태그 배포 가이드

## 📋 개요

메신저에서 링크를 공유할 때 예쁜 미리보기를 보여주기 위한 설정입니다.

## 🎯 목표

- ✅ 사이트 제목: "Momentum ETF"
- ✅ 설명: "추세추종 전략 기반 ETF 투자"
- ✅ 미리보기 이미지: 브랜드 컬러와 로고가 포함된 이미지

---

## 🔧 Nginx 설정

### 서버 접속 및 설정 파일 수정

```bash
# Oracle VM 서버 접속
ssh -i "~/DEV/ssh-key-2025-10-09.key" ubuntu@134.185.109.82

# Nginx 설정 파일 수정
sudo nano /etc/nginx/sites-available/etf.dojason.com
```

다음 내용을 추가:

```nginx
server {
    listen 80;
    server_name etf.dojason.com;

    # HTTPS로 리다이렉트 (Let's Encrypt 사용 시)
    # return 301 https://$server_name$request_uri;
# }

# server {
    # listen 443 ssl http2;
    # server_name etf.dojason.com;

    # SSL 인증서 (Let's Encrypt)
    # ssl_certificate /etc/letsencrypt/live/etf.dojason.com/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/etf.dojason.com/privkey.pem;

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

        # 버퍼 설정
        proxy_buffering off;
        proxy_read_timeout 86400;

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
</head>';
        sub_filter_once on;
        sub_filter_types text/html;
    }

    # 정적 파일 제공 (OG 이미지 등)
    location /static/ {
        alias /home/ubuntu/momentum-etf/static/;  # 실제 경로로 수정
        expires 30d;
        add_header Cache-Control "public, immutable";
        add_header Access-Control-Allow-Origin "*";
    }
}
```

### Nginx 재시작

```bash
# 설정 파일 문법 체크
sudo nginx -t

# 문제 없으면 재시작
sudo systemctl restart nginx

# 상태 확인
sudo systemctl status nginx
```

---

## ✅ 테스트

### 메타 태그 확인

```bash
curl https://etf.dojason.com/ | grep -i "og:title"
curl -I https://etf.dojason.com/static/og-image.png
```

### 온라인 도구로 테스트

1. **Facebook Sharing Debugger**
   - https://developers.facebook.com/tools/debug/
   - URL 입력: `https://etf.dojason.com/`
   - "Scrape Again" 버튼 클릭

2. **Twitter Card Validator**
   - https://cards-dev.twitter.com/validator
   - URL 입력 후 "Preview card" 클릭

3. **LinkedIn Post Inspector**
   - https://www.linkedin.com/post-inspector/
   - URL 입력 후 "Inspect" 클릭

4. **Open Graph Debugger**
   - https://www.opengraph.xyz/
   - 모든 메타 태그 확인 가능

---

## 🐛 문제 해결

### 문제 1: 메타 태그가 보이지 않음

```bash
# Nginx 로그 확인
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# sub_filter 모듈 확인
nginx -V 2>&1 | grep -o with-http_sub_module
```

### 문제 2: 이미지가 로드되지 않음

```bash
# 파일 권한 확인
ls -la /home/ubuntu/momentum-etf/static/og-image.png

# 권한 수정
chmod 644 /home/ubuntu/momentum-etf/static/og-image.png
```

### 문제 3: 캐시 문제

메신저 앱들은 링크 미리보기를 캐시합니다:
- Facebook: Sharing Debugger에서 "Scrape Again"
- Twitter: 24시간 후 자동 갱신
- 카카오톡: 캐시 클리어 불가, 시간이 지나면 갱신

---

## 📝 체크리스트

배포 전 확인사항:

- [ ] `static/og-image.png` 파일이 서버에 업로드됨
- [ ] Nginx 설정에 `sub_filter` 추가됨
- [ ] Nginx 설정에 `/static/` location 추가됨
- [ ] Nginx 재시작 완료
- [ ] `curl`로 메타 태그 확인
- [ ] 이미지 URL 접근 가능 확인
- [ ] Facebook Debugger로 테스트
- [ ] 실제 메신저에서 링크 공유 테스트

---

## 🎨 이미지 커스터마이징

더 나은 이미지를 원한다면:

1. **Canva** (https://www.canva.com/)
   - 템플릿: "Facebook Post" 또는 "Twitter Post"
   - 크기: 1200 x 630px
   - 브랜드 컬러: #D94D2B

2. **Figma** (https://www.figma.com/)
   - 프로페셔널한 디자인 가능

3. 디자이너에게 의뢰
   - 브랜드 아이덴티티 반영
   - 차트/그래프 포함

---

## 📚 참고 자료

- [Open Graph Protocol](https://ogp.me/)
- [Twitter Cards Documentation](https://developer.twitter.com/en/docs/twitter-for-websites/cards/overview/abouts-cards)
- [Facebook Sharing Best Practices](https://developers.facebook.com/docs/sharing/webmasters)
