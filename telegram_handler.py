from dotenv import load_dotenv
load_dotenv()

import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)
from chat_rag import (
    orchestrate_chat,
    get_or_create_chat_chain,
    classify_intent
)

# Environment variable for Telegram
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")
DOC_PATH = "uploads/company_doc.txt"  # Static document loaded from front-end config

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[BOT] Received /start command from user_id={update.effective_user.id}")
    await update.message.reply_text(
        "👋 Hello! I’m your Veridata AI Assistant.\n"
        "Ask me a business question based on your knowledge file, or just chat normally!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    user_id = str(update.effective_user.id)
    print(f"[BOT] Received message from user_id={user_id}: '{user_input}'")

    try:
        intent = classify_intent(user_input)
        print(f"[BOT] Intent classified as: {intent}")

        if intent == "business":
            print(f"[BOT] Running RAG for business query...")
            answer = orchestrate_chat(DOC_PATH, user_input)
            print(f"[BOT] RAG answer:\n{answer}")
        else:
            print(f"[BOT] Using memory chat chain...")
            chain = get_or_create_chat_chain(user_id)
            result = chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": user_id}}
            )
            answer = result.content if hasattr(result, "content") else str(result)
            print(f"[BOT] Memory chain answer:\n{answer}")

    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"
        print(f"[ERROR] Exception in handle_message: {e}")

    await update.message.reply_text(answer)

def main():
    print("[INIT] Starting Telegram bot...")
    if not TELEGRAM_API_KEY:
        raise ValueError("TELEGRAM_API_KEY not set. Please configure it in the .env file.")

    app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("[INIT] Bot is polling for messages.")
    app.run_polling()

if __name__ == "__main__":
    main()
