# Open Graph 메타 태그 설정

메신저 링크 미리보기를 위한 Open Graph 메타 태그 설정 가이드입니다.

## 완료된 설정

- ✅ BeautifulSoup로 Streamlit `index.html`에 메타 태그 자동 주입
- ✅ Docker 빌드 시 자동 적용
- ✅ 카카오톡, Facebook 테스트 성공

## 핵심 파일

1. **`add_meta_tags.py`** - BeautifulSoup로 Streamlit `index.html`에 메타 태그 주입
2. **`Dockerfile`** - Docker 빌드 시 스크립트 자동 실행
3. **`static/og-image.png`** - GitHub raw URL로 제공

## 작동 방식

```
Docker 빌드 → add_meta_tags.py 실행 → Streamlit index.html 수정 → 메타 태그 추가
```

**배포:** `upgrade` 브랜치에 push하면 GitHub Actions가 자동 배포

## 테스트

**카카오톡:** https://developers.kakao.com/tool/debugger/sharing ✅  
**Facebook:** https://developers.facebook.com/tools/debug/ ✅

---

## 문제 해결

**캐시 문제:**
- Facebook: Debugger에서 "다시 스크랩"
- 카카오톡: 24-48시간 후 자동 갱신

---

## 참고

- [Streamlit 앱 SEO 최적화](https://velog.io/@0o0w0d/Streamlit-%EC%95%B1-SEO-%EC%B5%9C%EC%A0%81%ED%99%94%ED%95%98%EA%B8%B0)
