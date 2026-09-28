"""강의 자료 RAG 텔레그램 봇.

- 과목을 고르고 PDF/PPTX를 보내면 인덱싱
- 그냥 질문하면 강의 자료를 근거로 답변 (출처 페이지 표시, 이어지는 질문 가능)
- /summary 로 파일 하나 요약, /quiz 로 퀴즈 풀기
"""

import asyncio
import functools
import html
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Poll, Update  # noqa: E402
from telegram.constants import ChatAction, ParseMode  # noqa: E402
from telegram.error import BadRequest  # noqa: E402
from telegram.ext import (  # noqa: E402
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PicklePersistence,
    filters,
)

from rag import llm  # noqa: E402
from rag.loader import SUPPORTED, chunk_file  # noqa: E402
from rag.store import LectureStore, get_embedder  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("lecture-bot")

ALLOWED_USER_IDS = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x}
HISTORY_TURNS = 3  # 이어지는 질문을 위해 기억할 최근 대화 수
TG_LIMIT = 4000  # 텔레그램 메시지 최대 길이(4096)보다 조금 작게

store = LectureStore()

HELP = """📚 <b>강의 자료 봇</b>

1. /subject 과목명 — 과목 선택 (예: <code>/subject 컴퓨터네트워크</code>)
2. PDF나 PPTX 파일을 보내면 그 과목에 저장돼요
3. 그냥 질문하면 강의 자료를 근거로 답해요

/subject — 과목 목록에서 고르기
/files — 현재 과목의 파일 목록 (삭제 가능)
/summary — 파일 하나를 골라 요약
/quiz [개수] — 강의 자료로 퀴즈 (기본 3문제)
/reset — 대화 기억 지우기"""


# ---------- 공통 유틸 ----------


def restricted(handler):
    """ALLOWED_USER_IDS에 있는 사람만 봇을 쓸 수 있게 한다 (API 요금 보호)."""

    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None or user.id not in ALLOWED_USER_IDS:
            if update.effective_message:
                await update.effective_message.reply_text(
                    f"허용되지 않은 사용자입니다.\n봇 주인이라면 .env의 ALLOWED_USER_IDS에 이 ID를 넣으세요: {user.id if user else '?'}"
                )
            return
        return await handler(update, context)

    return wrapper


def to_html(text: str) -> str:
    """Claude가 쓴 **굵게**, `코드`를 텔레그램 HTML로 바꾼다."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.S)
    return re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > limit // 2 else limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts + [text]


async def safe_edit(msg: Message, text: str, html_mode: bool = False) -> None:
    try:
        await msg.edit_text(text, parse_mode=ParseMode.HTML if html_mode else None)
    except BadRequest as e:
        if "not modified" in str(e):
            return
        if html_mode:  # HTML 파싱 실패 시 일반 텍스트로
            await msg.edit_text(re.sub(r"<[^>]+>", "", html.unescape(text)))
        else:
            raise


async def stream_reply(message: Message, chunks: AsyncIterator[str], footer: str = "") -> str:
    """Claude 답변을 받는 동안 메시지를 조금씩 수정해서 타이핑하듯 보여준다."""
    msg = await message.reply_text("💭 생각 중...")
    text, last_edit = "", 0.0
    try:
        async for piece in chunks:
            text += piece
            if time.monotonic() - last_edit > 1.5 and text.strip():
                await safe_edit(msg, text[-TG_LIMIT:] + " ▌")
                last_edit = time.monotonic()
    except anthropic.APIError as e:
        log.exception("Claude API error")
        await safe_edit(msg, f"⚠️ Claude API 오류: {getattr(e, 'message', e)}")
        return ""

    parts = split_message(to_html(text.strip()) + footer)
    await safe_edit(msg, parts[0], html_mode=True)
    for part in parts[1:]:
        await message.reply_text(part, parse_mode=ParseMode.HTML)
    return text


def format_sources(hits) -> str:
    pages: dict[str, set[int]] = defaultdict(set)
    for h in hits:
        pages[h.source].add(h.page)
    items = [f"{html.escape(src)} p.{', '.join(map(str, sorted(ps)))}" for src, ps in pages.items()]
    return "\n\n<i>📚 " + " · ".join(items) + "</i>"


def current_subject(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    return context.chat_data.get("subject")


async def need_subject(message: Message) -> None:
    await message.reply_text("먼저 과목을 골라주세요: /subject 과목명")


def set_subject(context: ContextTypes.DEFAULT_TYPE, subject: str) -> None:
    context.chat_data["subject"] = subject
    context.chat_data["history"] = []


# ---------- 명령어 ----------


@restricted
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = current_subject(context)
    now = f"\n\n현재 과목: <b>{html.escape(subject)}</b>" if subject else ""
    await update.message.reply_text(HELP + now, parse_mode=ParseMode.HTML)


@restricted
async def cmd_subject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        subject = " ".join(context.args).strip()[:20]  # 버튼 콜백 데이터 64바이트 제한
        set_subject(context, subject)
        n = store.subjects().get(subject, 0)
        await update.message.reply_text(
            f"📘 과목: {subject} (파일 {n}개)\n" + ("이제 질문하세요!" if n else "이 과목의 PDF/PPTX 파일을 보내주세요.")
        )
        return

    subjects = store.subjects()
    if not subjects:
        await update.message.reply_text("아직 과목이 없어요. /subject 과목명 으로 만들어 주세요.")
        return
    buttons = [[InlineKeyboardButton(f"{s} ({n})", callback_data=f"subj:{s}")] for s, n in subjects.items()]
    await update.message.reply_text("과목을 고르세요", reply_markup=InlineKeyboardMarkup(buttons))


@restricted
async def cmd_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = current_subject(context)
    if not subject:
        return await need_subject(update.message)
    files = store.files(subject)
    if not files:
        await update.message.reply_text(f"📘 {subject}: 아직 파일이 없어요. PDF/PPTX를 보내주세요.")
        return
    buttons = [[InlineKeyboardButton(f"🗑 {f}", callback_data=f"del:{i}")] for i, f in enumerate(files)]
    await update.message.reply_text(
        f"📘 {subject} 파일 {len(files)}개\n(버튼을 누르면 삭제돼요)", reply_markup=InlineKeyboardMarkup(buttons)
    )


@restricted
async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = current_subject(context)
    if not subject:
        return await need_subject(update.message)
    files = store.files(subject)
    if not files:
        await update.message.reply_text("요약할 파일이 없어요.")
        return
    buttons = [[InlineKeyboardButton(f, callback_data=f"sum:{i}")] for i, f in enumerate(files)]
    await update.message.reply_text("어떤 파일을 요약할까요?", reply_markup=InlineKeyboardMarkup(buttons))


@restricted
async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = current_subject(context)
    if not subject:
        return await need_subject(update.message)
    n = int(context.args[0]) if context.args and context.args[0].isdigit() else 3
    n = max(1, min(n, 10))

    hits = await asyncio.to_thread(store.sample, subject, n * 2)
    if not hits:
        await update.message.reply_text("퀴즈를 만들 자료가 없어요.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    status = await update.message.reply_text(f"📝 {subject} 퀴즈 {n}문제 만드는 중...")
    try:
        items = await llm.make_quiz(hits, n)
    except anthropic.APIError as e:
        log.exception("Claude API error")
        await safe_edit(status, f"⚠️ Claude API 오류: {getattr(e, 'message', e)}")
        return
    if not items:
        await safe_edit(status, "퀴즈를 만들지 못했어요. 다시 시도해 주세요.")
        return

    await status.delete()
    for i, q in enumerate(items, start=1):
        # 텔레그램 퀴즈 제한: 질문 300자, 보기 100자, 해설 200자
        await update.message.chat.send_poll(
            question=f"Q{i}. {q.question}"[:300],
            options=[c[:100] for c in q.choices[:10]],
            type=Poll.QUIZ,
            correct_option_id=q.answer_index,
            explanation=f"{q.explanation} ({q.source} p.{q.page})"[:200],
            is_anonymous=False,
        )


@restricted
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("🧹 대화 기억을 지웠어요.")


# ---------- 버튼 ----------


@restricted
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    action, _, value = query.data.partition(":")

    if action == "subj":
        set_subject(context, value)
        await query.edit_message_text(f"📘 과목: {value}\n이제 질문하세요!")
        return

    subject = current_subject(context)
    files = store.files(subject) if subject else []
    if not value.isdigit() or int(value) >= len(files):
        await query.edit_message_text("파일 목록이 바뀌었어요. 다시 시도해 주세요.")
        return
    source = files[int(value)]

    if action == "del":
        n = await asyncio.to_thread(store.delete_file, subject, source)
        await query.edit_message_text(f"🗑 {source} 삭제 ({n}개 청크)")
    elif action == "sum":
        await query.edit_message_text(f"📄 {source} 요약")
        pages = await asyncio.to_thread(store.file_text, subject, source)
        await stream_reply(query.message, llm.summarize(source, pages))


# ---------- 파일 업로드 ----------


@restricted
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    ext = Path(doc.file_name or "").suffix.lower()
    if ext not in SUPPORTED:
        await update.message.reply_text("PDF나 PPTX 파일만 올릴 수 있어요. (PPT는 PPTX로 저장해서 보내주세요)")
        return
    if doc.file_size and doc.file_size > 20 * 1024 * 1024:
        await update.message.reply_text("텔레그램 봇은 20MB가 넘는 파일을 받을 수 없어요. 파일을 나눠서 보내주세요.")
        return

    # 캡션에 과목명을 적으면 그 과목으로 저장
    if update.message.caption:
        set_subject(context, update.message.caption.strip()[:20])
    subject = current_subject(context)
    if not subject:
        await update.message.reply_text("어느 과목 자료인가요? /subject 과목명 으로 먼저 고르거나, 파일 캡션에 과목명을 적어주세요.")
        return

    status = await update.message.reply_text(f"📥 {doc.file_name} 처리 중...")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{ext}"
        await (await doc.get_file()).download_to_drive(path)
        try:
            chunks = await asyncio.to_thread(chunk_file, path, subject, doc.file_name)
        except Exception:
            log.exception("failed to read %s", doc.file_name)
            await safe_edit(status, "⚠️ 파일을 읽지 못했어요. 암호가 걸렸거나 손상된 파일인지 확인해 주세요.")
            return

    if not chunks:
        await safe_edit(status, "⚠️ 텍스트를 찾지 못했어요. 스캔한 이미지 PDF는 아직 지원하지 않아요.")
        return

    store.delete_file(subject, doc.file_name)  # 같은 이름의 파일을 다시 올리면 교체
    await asyncio.to_thread(store.add, chunks)
    pages = len({c.page for c in chunks})
    await safe_edit(status, f"✅ [{subject}] {doc.file_name}\n{pages}페이지, {len(chunks)}개 청크 저장 완료. 질문하세요!")


# ---------- 질문 ----------


@restricted
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = current_subject(context)
    if not subject:
        return await need_subject(update.message)

    question = update.message.text
    history: list[dict] = context.chat_data.setdefault("history", [])

    # "그거 더 자세히" 같은 이어지는 질문도 검색되도록 직전 질문을 붙여서 찾는다
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    await update.message.chat.send_action(ChatAction.TYPING)
    hits = await asyncio.to_thread(store.search, subject, f"{prev} {question}".strip(), 5)
    if not hits:
        await update.message.reply_text(f"📘 {subject}에 저장된 자료가 없어요. PDF/PPTX를 먼저 보내주세요.")
        return

    reply = await stream_reply(update.message, llm.answer(question, hits, history), footer=format_sources(hits))
    if reply:
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        del history[: -HISTORY_TURNS * 2]


# ---------- 실행 ----------


async def post_init(app: Application) -> None:
    await asyncio.to_thread(get_embedder)  # 첫 질문이 느리지 않게 임베딩 모델을 미리 로드
    await app.bot.set_my_commands(
        [
            ("subject", "과목 선택"),
            ("files", "파일 목록 / 삭제"),
            ("summary", "파일 요약"),
            ("quiz", "퀴즈 풀기"),
            ("reset", "대화 기억 지우기"),
            ("help", "사용법"),
        ]
    )
    log.info("봇 시작 (허용 사용자: %s)", ALLOWED_USER_IDS or "없음 - /start 로 내 ID를 확인하세요")


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(".env에 TELEGRAM_BOT_TOKEN을 넣어주세요. (README 참고)")

    Path("data").mkdir(exist_ok=True)
    app = (
        Application.builder()
        .token(token)
        .persistence(PicklePersistence(filepath="data/bot_state.pickle"))  # 재시작해도 현재 과목 유지
        .concurrent_updates(True)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("subject", cmd_subject))
    app.add_handler(CommandHandler("files", cmd_files))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.run_polling()


if __name__ == "__main__":
    main()
