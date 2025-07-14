
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from rag import qa_chain
import os

telegram_key = os.getenv("TELEGRAM_API_KEY")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am your AI assistant. Ask me anything about AcmeTech Solutions.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    try:
        rag_response = qa_chain.invoke(user_input)
        response = rag_response["result"]
    except Exception as e:
        response = f"⚠️ An error occurred while processing your request: {e}"

    await update.message.reply_text(response)

if __name__ == "__main__":
    app = ApplicationBuilder().token(telegram_key).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()
