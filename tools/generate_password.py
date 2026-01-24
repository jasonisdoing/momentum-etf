"""
Streamlit Authenticator용 비밀번호 해시 생성 및 검증 도구입니다.

[사용법]
1. 해시 생성 (기본):
   python tools/generate_password.py

2. 해시 검증 (비밀번호와 해시가 일치하는지 확인):
   python tools/generate_password.py --verify

[주의]
이 스크립트를 실행하려면 streamlit-authenticator 패키지가 설치되어 있어야 합니다.
(pip install streamlit-authenticator)
"""

import argparse
import getpass
import sys

try:
    import streamlit_authenticator as stauth
except ImportError:
    print("오류: 'streamlit-authenticator' 패키지가 설치되어 있지 않습니다.")
    print("설치 명령: pip install streamlit-authenticator")
    sys.exit(1)


def generate_hash():
    print("\n=== 비밀번호 해시 생성기 ===")
    password = getpass.getpass("비밀번호 입력: ")
    confirm = getpass.getpass("비밀번호 확인: ")

    if password != confirm:
        print("\n[오류] 비밀번호가 일치하지 않습니다. 다시 시도해주세요.")
        return

    if not password:
        print("\n[오류] 비밀번호가 비어있습니다.")
        return

    # 해시 생성
    hashed_passwords = stauth.Hasher([password]).generate()
    hashed = hashed_passwords[0]

    print(f"\n[성공] 생성된 해시값:\n{hashed}")
    print("\n위 값을 .streamlit/secrets.toml 파일의 'password' 항목에 복사하여 붙여넣으세요.")


def verify_hash():
    print("\n=== 비밀번호 검증기 ===")
    password = getpass.getpass("확인할 비밀번호 입력: ")
    hashed_input = input("비교할 해시값 입력: ").strip()

    if not password or not hashed_input:
        print("\n[오류] 비밀번호와 해시값을 모두 입력해야 합니다.")
        return

    # 검증 (새로운 Hasher 인스턴스 사용 방식 대신 직접 check 호출 방식이 있는지 확인 필요하지만,
    # stauth 버전에 따라 다르므로 가장 확실한 방법은 check_pw 함수가 있다면 사용,
    # 없다면 Hasher 내부 로직 사용. 여기서는 Hasher의 check_pw 메서드 사용)

    # streamlit-authenticator 버전에 따라 API가 다를 수 있음.
    # 최신 버전(0.3.x 이상)에서는 Hasher.check_pw() 사용 가능 여부 확인
    try:
        # bcrypt 라이브러리를 직접 사용하여 검증하는 것이 가장 확실함 (stauth 의존성 줄임)
        import bcrypt

        # 입력된 해시가 바이트가 아니면 인코딩
        if isinstance(hashed_input, str):
            hashed_bytes = hashed_input.encode("utf-8")
        else:
            hashed_bytes = hashed_input

        password_bytes = password.encode("utf-8")

        is_correct = bcrypt.checkpw(password_bytes, hashed_bytes)

        if is_correct:
            print("\n[일치] 비밀번호가 해시값과 일치합니다. ✅")
        else:
            print("\n[불일치] 비밀번호가 일치하지 않습니다. ❌")

    except Exception as e:
        print(f"\n[오류] 검증 중 에러 발생: {e}")
        print("해시값이 올바른 bcrypt 형식인지 확인해주세요.")


def main():
    parser = argparse.ArgumentParser(description="비밀번호 해시 생성 및 검증 도구")
    parser.add_argument("-v", "--verify", action="store_true", help="해시값 검증 모드 실행")

    args = parser.parse_args()

    if args.verify:
        verify_hash()
    else:
        generate_hash()


if __name__ == "__main__":
    main()
