# 🚀 Open Graph 메타 태그 배포 - 빠른 가이드

## ✅ 준비 완료
- [x] OG 이미지 생성됨: `static/og-image.png` (1200x630px)
- [x] 배포 가이드 작성됨: `docs/deploy-og-tags.md`

---

## 📦 Nginx 설정 (1회만 필요)

> **참고:** 코드 변경은 `upgrade` 브랜치에 push하면 GitHub Actions가 자동 배포합니다.

### Nginx 설정 추가
```bash
# Oracle VM 서버 접속
ssh -i "~/DEV/ssh-key-2025-10-09.key" ubuntu@134.185.109.82

# Nginx 설정 파일 수정
sudo nano /etc/nginx/sites-available/etf.dojason.com
```

**추가할 내용:**
```nginx
location / {
    proxy_pass http://localhost:8501;
    # ... 기존 proxy 설정 ...
    
    # 메타 태그 주입 (이 부분 추가)
    sub_filter '</head>' '
    <meta property="og:title" content="Momentum ETF" />
    <meta property="og:description" content="추세추종 전략 기반 ETF 투자" />
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
    alias /home/ubuntu/apps/momentum-etf/static/;
    expires 30d;
    add_header Cache-Control "public, immutable";
}
```

### Nginx 재시작
```bash
sudo nginx -t
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
- **제목**: "Momentum ETF"
- **설명**: "추세추종 전략 기반 ETF 투자"
- **이미지**: 브랜드 컬러(#D94D2B)가 포함된 1200x630 이미지

---

## 🐛 문제 해결

**메타 태그가 안 보이면:**
```bash
sudo tail -f /var/log/nginx/error.log
```

**이미지가 안 보이면:**
```bash
chmod 644 /home/ubuntu/apps/momentum-etf/static/og-image.png
```

**캐시 문제:**
- Facebook: Debugger에서 "Scrape Again"
- 카카오톡: 시간이 지나면 자동 갱신 (즉시 불가)

---

## 📚 상세 가이드
- 전체 가이드: `docs/deploy-og-tags.md`
- Nginx 설정: `docs/nginx-meta-tags-setup.md`
