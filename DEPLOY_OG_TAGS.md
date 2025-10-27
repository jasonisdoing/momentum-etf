# 🚀 Open Graph 메타 태그 배포 - 빠른 가이드

## ✅ 준비 완료
- [x] OG 이미지 생성됨: `static/og-image.png` (1200x630px)
- [x] 배포 가이드 작성됨: `docs/deploy-og-tags.md`

---

## 📦 서버 배포 3단계

### 1️⃣ 파일 업로드
```bash
# 서버로 static 폴더 업로드
scp -r static/ user@etf.dojason.com:/home/ubuntu/momentum-etf/
```

### 2️⃣ Nginx 설정 수정
```bash
ssh user@etf.dojason.com
sudo nano /etc/nginx/sites-available/etf.dojason.com
```

**추가할 내용:**
```nginx
location / {
    proxy_pass http://localhost:8501;
    # ... 기존 proxy 설정 ...
    
    # 메타 태그 주입 (이 부분 추가)
    sub_filter '</head>' '
    <meta property="og:title" content="Momentum ETF - 모멘텀 투자 전략 대시보드" />
    <meta property="og:description" content="데이터 기반 모멘텀 투자 전략으로 포트폴리오를 관리하세요. 실시간 추천 및 성과 분석을 제공합니다." />
    <meta property="og:image" content="https://etf.dojason.com/static/og-image.png" />
    <meta property="og:url" content="https://etf.dojason.com/" />
    <meta property="og:type" content="website" />
    <meta property="og:site_name" content="Momentum ETF" />
    <meta name="twitter:card" content="summary_large_image" />
</head>';
    sub_filter_once on;
    sub_filter_types text/html;
}

# 정적 파일 제공 (이 블록 추가)
location /static/ {
    alias /home/ubuntu/momentum-etf/static/;
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

### 3️⃣ Nginx 재시작
```bash
sudo nginx -t          # 설정 테스트
sudo systemctl restart nginx
```

---

## 🧪 테스트

### 빠른 테스트
```bash
# 메타 태그 확인
curl https://etf.dojason.com/ | grep "og:title"

# 이미지 확인
curl -I https://etf.dojason.com/static/og-image.png
```

### 온라인 테스트
1. **Facebook**: https://developers.facebook.com/tools/debug/
2. **Twitter**: https://cards-dev.twitter.com/validator
3. **카카오톡**: 직접 링크 공유해보기

---

## 📝 예상 결과

메신저에서 링크 공유 시:
- **제목**: "Momentum ETF - 모멘텀 투자 전략 대시보드"
- **설명**: "데이터 기반 모멘텀 투자 전략으로..."
- **이미지**: 브랜드 컬러(#D94D2B)가 포함된 1200x630 이미지

---

## 🐛 문제 해결

**메타 태그가 안 보이면:**
```bash
sudo tail -f /var/log/nginx/error.log
```

**이미지가 안 보이면:**
```bash
chmod 644 /home/ubuntu/momentum-etf/static/og-image.png
```

**캐시 문제:**
- Facebook: Debugger에서 "Scrape Again"
- 카카오톡: 시간이 지나면 자동 갱신 (즉시 불가)

---

## 📚 상세 가이드
- 전체 가이드: `docs/deploy-og-tags.md`
- Nginx 설정: `docs/nginx-meta-tags-setup.md`
