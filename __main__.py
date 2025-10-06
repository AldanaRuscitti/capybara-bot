import io
import json
import logging
import os
import unicodedata
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, InputFile, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, ConversationHandler, ContextTypes, filters, CallbackQueryHandler
import pandas as pd

try:
    import psycopg2
except ImportError:  # pragma: no cover - optional dependency
    psycopg2 = None

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise RuntimeError('Missing TELEGRAM_BOT_TOKEN environment variable')

CONFIG_PATH = os.getenv('BOT_CONFIG_PATH', 'config.json')
DATA_DIR = os.getenv('BOT_DATA_DIR', '.')
os.makedirs(DATA_DIR, exist_ok=True)


def load_config(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as config_file:
            return json.load(config_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f'Configuration file not found at {path}') from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'Invalid JSON in configuration file {path}: {exc}') from exc


CONFIG = load_config(CONFIG_PATH)

DATABASE_URL = os.getenv('DATABASE_URL')
DB_ENABLED = bool(DATABASE_URL and psycopg2)


def db_connection():
    if not DB_ENABLED:
        raise RuntimeError('Database connection requested but DATABASE_URL/psycopg2 not available')
    return psycopg2.connect(DATABASE_URL)


def init_db() -> None:
    if not DB_ENABLED:
        return
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS movements (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT NOT NULL,
                    numero_movimiento INTEGER NOT NULL,
                    movement_type TEXT,
                    amount DOUBLE PRECISION,
                    currency TEXT,
                    description TEXT,
                    payment_method TEXT,
                    comment TEXT,
                    fecha TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS balances (
                    chat_id BIGINT NOT NULL,
                    cuenta TEXT NOT NULL,
                    saldo DOUBLE PRECISION NOT NULL,
                    fecha_actualizacion TIMESTAMP NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (chat_id, cuenta)
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_movements_chat_fecha
                    ON movements (chat_id, fecha DESC);
                """
            )
        conn.commit()


def save_config() -> None:
    with open(CONFIG_PATH, 'w', encoding='utf-8') as config_file:
        json.dump(CONFIG, config_file, indent=2, ensure_ascii=False)


def refresh_config_cache() -> None:
    global ACCOUNTS, CATEGORIES, PAYMENT_METHODS_CONFIG, INCOME_CATEGORIES, EXPENSE_CATEGORIES
    global PAYMENT_METHOD_NAMES, PAYMENT_METHOD_LOOKUP, PAYMENT_ACCOUNT_MAP, NON_ADJUST_ACCOUNTS

    ACCOUNTS = CONFIG.setdefault('accounts', [])
    CATEGORIES = CONFIG.setdefault('categories', {})
    PAYMENT_METHODS_CONFIG = CONFIG.setdefault('payment_methods', [])
    NON_ADJUST_ACCOUNTS = set(CONFIG.get('non_adjust_accounts', []))

    if not ACCOUNTS:
        raise RuntimeError('Configuration must include a non-empty "accounts" list')

    if 'income' not in CATEGORIES or 'expense' not in CATEGORIES:
        raise RuntimeError('Configuration must include "categories.income" and "categories.expense" entries')

    INCOME_CATEGORIES = CATEGORIES['income']
    EXPENSE_CATEGORIES = CATEGORIES['expense']

    PAYMENT_METHOD_NAMES = [entry['name'] for entry in PAYMENT_METHODS_CONFIG if entry.get('name')]
    if not PAYMENT_METHOD_NAMES:
        raise RuntimeError('Configuration must include at least one payment method with a "name"')

    PAYMENT_METHOD_LOOKUP = {
        entry['name'].lower(): entry
        for entry in PAYMENT_METHODS_CONFIG
        if entry.get('name')
    }
    PAYMENT_ACCOUNT_MAP = {
        entry['name']: entry['account']
        for entry in PAYMENT_METHODS_CONFIG
        if entry.get('name') and entry.get('account')
    }


refresh_config_cache()


def get_chat_data_dir(chat_id: int) -> str:
    path = os.path.join(DATA_DIR, str(chat_id))
    os.makedirs(path, exist_ok=True)
    return path


def finanzas_path(chat_id: int) -> str:
    return os.path.join(get_chat_data_dir(chat_id), 'finanzas.csv')


def saldos_path(chat_id: int) -> str:
    return os.path.join(get_chat_data_dir(chat_id), 'saldos.csv')


def load_finanzas_dataframe(chat_id: int) -> pd.DataFrame:
    if DB_ENABLED:
        with db_connection() as conn:
            query = (
                "SELECT chat_id, numero_movimiento, movement_type, amount, currency, "
                "description, payment_method, comment, fecha "
                "FROM movements WHERE chat_id = %s ORDER BY fecha DESC"
            )
            df = pd.read_sql_query(query, conn, params=(chat_id,))
            return df
    path = finanzas_path(chat_id)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        columns = [
            "chat_id",
            "numero_movimiento",
            "movement_type",
            "amount",
            "currency",
            "description",
            "payment_method",
            "comment",
            "fecha",
        ]
        return pd.DataFrame(columns=columns)

    if not df.empty:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['fecha'])
        df = df.sort_values(by='fecha', ascending=False)
    return df


def load_saldos_dataframe(chat_id: int) -> pd.DataFrame:
    if DB_ENABLED:
        with db_connection() as conn:
            query = (
                "SELECT chat_id, cuenta, saldo, fecha_actualizacion "
                "FROM balances WHERE chat_id = %s"
            )
            return pd.read_sql_query(query, conn, params=(chat_id,))
    path = saldos_path(chat_id)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["chat_id", "cuenta", "saldo", "fecha_actualizacion"])

    if not df.empty:
        df['saldo'] = pd.to_numeric(df['saldo'], errors='coerce')
        df['fecha_actualizacion'] = pd.to_datetime(df['fecha_actualizacion'], errors='coerce')
    return df


def format_currency(value: float, currency_label: str = '$') -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return f"{currency_label}0.00"
    return f"{currency_label}{amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')


def parse_date(value: str) -> Optional[datetime]:
    value = value.strip()
    for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def classify_account(account: str) -> str:
    if not account:
        return 'unknown'

    lowered = unicodedata.normalize('NFKD', account).encode('ascii', 'ignore').decode('ascii').lower()

    if account in NON_ADJUST_ACCOUNTS or any(keyword in lowered for keyword in ('cocos', 'binance', 'bull', 'lemon')):
        return 'investment'
    if any(keyword in lowered for keyword in ('usd', 'dolar', 'dólar', 'u$s')):
        return 'usd'
    return 'ars'


def attach_account_info(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df['account'] = df['payment_method'].apply(lambda method: resolve_payment_method(method)[0])
    return df


def filter_movements(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = attach_account_info(df)

    if filters.get('from_date'):
        filtered = filtered[filtered['fecha'] >= filters['from_date']]
    if filters.get('to_date'):
        filtered = filtered[filtered['fecha'] <= filters['to_date']]
    if filters.get('account'):
        account_match = find_account(filters['account']) or filters['account']
        filtered = filtered[filtered['account'].fillna('').str.lower() == account_match.lower()]
    if filters.get('movement_type'):
        filtered = filtered[filtered['movement_type'] == filters['movement_type']]
    if filters.get('description'):
        target = filters['description'].lower()
        filtered = filtered[filtered['description'].fillna('').str.lower().str.contains(target)]
    if filters.get('payment_method'):
        target = filters['payment_method'].lower()
        filtered = filtered[filtered['payment_method'].fillna('').str.lower().str.contains(target)]
    if filters.get('comment_query'):
        filtered = filtered[filtered['comment'].fillna('').str.contains(filters['comment_query'], case=False, na=False)]

    return filtered.sort_values(by='fecha', ascending=False)


def parse_filter_text(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    parts = [item.strip() for item in text.split(';') if item.strip()]

    for part in parts:
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")

        if key in {'desde', 'from', 'from_date'}:
            parsed = parse_date(value)
            if parsed:
                result['from_date'] = parsed
        elif key in {'hasta', 'to', 'to_date'}:
            parsed = parse_date(value)
            if parsed:
                result['to_date'] = parsed + timedelta(days=1) - timedelta(seconds=1)
        elif key in {'cuenta', 'account'}:
            if value:
                result['account'] = value
        elif key in {'tipo', 'type'}:
            lowered = value.lower()
            if 'ing' in lowered:
                result['movement_type'] = '🤑 Ingreso'
            elif 'gas' in lowered:
                result['movement_type'] = '🚨 Gasto'
        elif key in {'categoria', 'description', 'subcategoria'}:
            if value:
                result['description'] = value
        elif key in {'metodo', 'pago', 'payment', 'payment_method'}:
            if value:
                result['payment_method'] = value
        elif key in {'comentario', 'comment', 'busca'}:
            if value:
                result['comment_query'] = value

    return result


def describe_movement(row: pd.Series) -> str:
    fecha = row['fecha']
    fecha_text = fecha.strftime('%d/%m/%Y %H:%M') if pd.notnull(fecha) else 'Sin fecha'
    amount = row['amount'] if pd.notnull(row['amount']) else 0.0
    sign = '-' if row['movement_type'] == '🚨 Gasto' else '+'
    amount_text = f"{sign}${abs(amount):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    description = row.get('description', 'Sin descripción')
    payment = row.get('payment_method', 'Sin medio de pago')
    comment = row.get('comment')
    comment_text = f"\n📝 {comment}" if isinstance(comment, str) and comment.strip() else ''
    account = row.get('account') or ''
    account_text = f" ({account})" if account else ''

    return (
        f"📅 {fecha_text}\n"
        f"{amount_text} - {row['movement_type']}\n"
        f"🏷️ {description}\n"
        f"💳 {payment}{account_text}{comment_text}"
    )


def build_movements_message(df: pd.DataFrame, limit: int = 10) -> str:
    if df.empty:
        return 'No se encontraron movimientos con esos criterios.'

    total = len(df)
    subset = df.head(limit)
    rows = subset.apply(describe_movement, axis=1)
    header = f"Mostrando {len(subset)} de {total} movimientos encontrados:\n\n" if total > 1 else ''
    return header + '\n\n'.join(rows.tolist())


def compute_account_totals(df_saldos: pd.DataFrame) -> Dict[str, float]:
    totals = {'ars': 0.0, 'usd': 0.0, 'investment': 0.0, 'otros': 0.0}

    if df_saldos.empty:
        return totals

    for _, row in df_saldos.iterrows():
        cuenta = row.get('cuenta')
        saldo = row.get('saldo')
        if pd.isna(saldo):
            continue
        category = classify_account(str(cuenta))
        if category not in totals:
            totals['otros'] += float(saldo)
        else:
            totals[category] += float(saldo)
    return totals


def compute_monthly_totals(df_finanzas: pd.DataFrame, reference: Optional[datetime] = None) -> Dict[str, float]:
    if df_finanzas.empty:
        return {'ingresos': 0.0, 'gastos': 0.0}

    if reference is None:
        reference = datetime.now()

    start_month = reference.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = (start_month + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)

    current_month = df_finanzas[(df_finanzas['fecha'] >= start_month) & (df_finanzas['fecha'] <= end_month)]

    totals = {
        'ingresos': current_month[current_month['movement_type'] == '🤑 Ingreso']['amount'].sum(),
        'gastos': current_month[current_month['movement_type'] == '🚨 Gasto']['amount'].sum(),
    }

    totals['gastos'] = abs(totals['gastos']) if pd.notnull(totals['gastos']) else 0.0
    totals['ingresos'] = totals['ingresos'] if pd.notnull(totals['ingresos']) else 0.0
    return totals


def generate_expense_chart(df_finanzas: pd.DataFrame) -> Optional[io.BytesIO]:
    if df_finanzas.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    gastos = df_finanzas[df_finanzas['movement_type'] == '🚨 Gasto'].copy()
    if gastos.empty:
        return None

    gastos['month'] = gastos['fecha'].dt.to_period('M')
    ref_period = pd.Period(datetime.now(), freq='M')
    gastos = gastos[gastos['month'] == ref_period]

    if gastos.empty:
        return None

    aggregated = gastos.groupby('description')['amount'].sum().sort_values(ascending=False).head(8)
    aggregated = aggregated.abs()

    if aggregated.empty:
        return None

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aggregated.plot(kind='barh', ax=ax, color='#ff7043')
    ax.set_xlabel('Monto (ARS)')
    ax.set_ylabel('Categoría')
    ax.set_title('Top gastos del mes')
    ax.invert_yaxis()
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    return buffer


def build_summary_message(totals: Dict[str, float], month_totals: Dict[str, float]) -> str:
    ars_text = format_currency(totals.get('ars', 0.0))
    usd_text = format_currency(totals.get('usd', 0.0), 'USD ')
    inv_text = format_currency(totals.get('investment', 0.0))
    other_text = format_currency(totals.get('otros', 0.0)) if totals.get('otros', 0.0) else None

    ingresos_text = format_currency(month_totals.get('ingresos', 0.0))
    gastos_text = format_currency(month_totals.get('gastos', 0.0))
    balance = month_totals.get('ingresos', 0.0) - month_totals.get('gastos', 0.0)
    balance_text = format_currency(balance)

    lines = [
        '📊 *Resumen financiero*',
        '',
        f"💵 Saldos en pesos: {ars_text}",
        f"💵 Saldos en dólares: {usd_text}",
        f"📈 Inversiones: {inv_text}",
    ]

    if other_text:
        lines.append(f"🔁 Otros: {other_text}")

    lines.extend([
        '',
        '📆 *Mes en curso*',
        f"🟢 Ingresos: {ingresos_text}",
        f"🔴 Gastos: {gastos_text}",
        f"⚖️ Balance: {balance_text}",
    ])

    return '\n'.join(lines)


HELP_MESSAGE = (
    "ℹ️ *Centro de ayuda Capy*\n\n"
    "Capybara te ayuda a registrar movimientos, controlar saldos y revisar tus finanzas sin salir de Telegram.\n\n"
    "*Comandos esenciales*\n"
    "• /inicio — mostrar el menú principal\n"
    "• /ultimos — explorar movimientos recientes con filtros\n"
    "• /resumen — resumen de saldos, ingresos y gastos del mes\n"
    "• /help — volver a esta guía\n\n"
    "*Tips rápidos*\n"
    "• Guardá un comentario en cada movimiento para buscarlo después (`comentario=...`).\n"
    "• Personalizá cuentas, categorías y métodos con /add_account, /add_category y /add_payment_method.\n"
    "• Volvé al menú cuando quieras con /inicio o el botón 'ℹ️ Ayuda'."
)


def detect_currency_symbol(currency: Optional[str]) -> str:
    if not currency:
        return '$'
    lowered = unicodedata.normalize('NFKD', currency).encode('ascii', 'ignore').decode('ascii').lower()
    if 'usd' in lowered or 'dolar' in lowered or 'dólar' in currency.lower():
        return 'USD '
    if 'cripto' in lowered or 'btc' in lowered or 'crypto' in lowered:
        return '₿ '
    return '$'


def build_movement_confirmation(data: Dict[str, Any]) -> str:
    movement_type = data.get('movement_type', '')
    raw_amount = str(data.get('amount', '0')).replace(',', '.').strip()
    try:
        amount_value = float(raw_amount)
    except ValueError:
        amount_value = 0.0
    symbol = detect_currency_symbol(data.get('currency'))
    amount_text = format_currency(amount_value, symbol)
    description = data.get('description', 'Sin categoría')
    payment = data.get('payment_method', 'Sin medio de pago')
    comment = data.get('comment')
    comment_line = f"\n📝 {comment}" if comment else ''

    icon = '🤑' if movement_type == '🤑 Ingreso' else '🚨'

    return (
        f"✅ Movimiento guardado\n"
        f"{icon} {movement_type} de {amount_text}\n"
        f"🏷️ {description}\n"
        f"💳 {payment}{comment_line}"
    )


def format_account_balance(account: str, saldo: Any) -> str:
    try:
        value = float(saldo)
    except (TypeError, ValueError):
        return str(saldo)

    category = classify_account(account)
    symbol = 'USD ' if category == 'usd' else '$'
    return format_currency(value, symbol)
def build_keyboard(options, items_per_row=2):
    if not options:
        return [[]]
    return [options[i:i + items_per_row] for i in range(0, len(options), items_per_row)]


def make_account_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(build_keyboard(ACCOUNTS), one_time_keyboard=True, resize_keyboard=True)


def make_yes_no_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([['Sí', 'No']], one_time_keyboard=True, resize_keyboard=True)


def description_keyboard(movement_type: str) -> InlineKeyboardMarkup:
    options = INCOME_CATEGORIES if movement_type == '🤑 Ingreso' else EXPENSE_CATEGORIES
    buttons = [[InlineKeyboardButton(item['text'], callback_data=item['value'])] for item in options]
    return InlineKeyboardMarkup(buttons)


def payment_method_keyboard() -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(name, callback_data=name)] for name in PAYMENT_METHOD_NAMES]
    return InlineKeyboardMarkup(buttons)


def slugify(text: str) -> str:
    normalized = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    normalized = normalized.lower().strip()
    sanitized = ''.join(char if char.isalnum() else '_' for char in normalized)
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    sanitized = sanitized.strip('_')
    return sanitized or 'item'


def infer_account_from_name(method_name: str) -> Optional[str]:
    if not method_name:
        return None

    normalized = unicodedata.normalize('NFKD', method_name).encode('ascii', 'ignore').decode('ascii').lower()

    if 'naranja' in normalized:
        return 'Naranja'
    tokens = normalized.split()

    if 'mercado pago' in normalized or ('mercado' in tokens and 'pago' in tokens) or 'mp' in tokens:
        return 'Mercado Pago'
    if 'uala' in normalized:
        return 'UALA'
    if 'galicia' in normalized:
        if 'usd' in normalized or 'dolar' in normalized or 'dólar' in method_name.lower():
            return 'CA USD Galicia'
        return 'CA $ Galicia'
    if 'efectivo' in normalized:
        if 'usd' in normalized or 'dolar' in normalized:
            return 'Efectivo Dólares'
        return 'Efectivo Pesos'
    if 'binance' in normalized:
        return 'Binance'
    if 'cocos' in normalized:
        return 'Cocos'
    if 'bull' in normalized:
        return 'Bull Market'
    if 'lemon' in normalized:
        return 'Lemon'

    return None


def resolve_payment_method(method_name: str) -> Tuple[Optional[str], bool]:
    if not method_name:
        return None, False

    entry = PAYMENT_METHOD_LOOKUP.get(method_name.lower())
    account = entry.get('account') if entry else None
    adjust = entry.get('adjust_balance') if entry and 'adjust_balance' in entry else None

    if not account:
        account = infer_account_from_name(method_name)

    if adjust is None:
        adjust = account not in NON_ADJUST_ACCOUNTS if account else False

    return account, bool(adjust)


def find_account(name: str) -> Optional[str]:
    lower_name = name.lower()
    for account in ACCOUNTS:
        if account.lower() == lower_name:
            return account
    return None


def payment_method_exists(name: str) -> bool:
    return name.lower() in PAYMENT_METHOD_LOOKUP

# Configuración de alertas
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Variables
(ACTION_TYPE, MOVEMENT_TYPE, AMOUNT, CURRENCY, DESCRIPTION, PAYMENT_METHOD, COMMENT, CONTINUE, UPDATE_ACCOUNT, CONSULT_ACCOUNT, UPDATE_AMOUNT, ULTIMOS_CHOICE, ULTIMOS_FILTER) = range(13)

MAIN_MENU_LAYOUT = [
    ['➕ Registrar movimiento', '📋 Últimos movimientos'],
    ['💰 Ajustar saldos', '🔍 Consultar saldos'],
    ['📊 Resumen Capy', 'ℹ️ Ayuda']
]

WELCOME_TEMPLATE = (
    "🌿 *Capybara Finanzas*\n"
    "¡Hola {first_name}! Soy Capy, tu copiloto financiero.\n\n"
    "Elegí una opción del menú o probá estos atajos:\n"
    "• /ultimos — últimos movimientos con filtros inteligentes\n"
    "• /resumen — estado general, ingresos y gastos del mes\n"
    "• /help — guía completa para aprovechar Capy"
)

BOT_COMMANDS_CONFIG = [
    BotCommand('inicio', 'Mostrar el menú principal'),
    BotCommand('movimientos', 'Consultar últimos movimientos'),
    BotCommand('resumen', 'Ver resumen financiero'),
    BotCommand('ayuda', 'Ver ayuda y comandos disponibles'),
    BotCommand('agregar_cuenta', 'Registrar una cuenta'),
    BotCommand('agregar_categoria', 'Registrar una categoría'),
    BotCommand('agregar_medio_pago', 'Registrar un medio de pago'),
]


def make_main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(MAIN_MENU_LAYOUT, resize_keyboard=True)


def build_welcome_text(user) -> str:
    first_name = getattr(user, 'first_name', None) or 'amig@'
    return WELCOME_TEMPLATE.format(first_name=first_name)

# Inicio
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        build_welcome_text(update.effective_user),
        parse_mode='Markdown',
        reply_markup=make_main_menu_keyboard()
    )
    return ACTION_TYPE

# Elección entre agregar movimiento, actualizar o consultar saldos
async def action_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_choice = update.message.text

    if user_choice in {'➕ Registrar movimiento', '➕ Registrar Movimiento', '➕ Agregar Movimiento'}:
        context.user_data['action_type'] = 'Agregar Movimiento'
        await update.message.reply_text(
            "¿Qué tipo de movimiento querés registrar?",
            parse_mode='Markdown',
            reply_markup=ReplyKeyboardMarkup([['🤑 Ingreso', '🚨 Gasto']], one_time_keyboard=True, resize_keyboard=True)
        )
        return MOVEMENT_TYPE

    elif user_choice in {'💰 Ajustar saldos', '💰 Ajustar Saldos', '💰 Actualizar Saldos'}:
        context.user_data['action_type'] = 'Actualizar Saldos'
        await update.message.reply_text(
            "Elegí la cuenta que querés actualizar:",
            reply_markup=make_account_keyboard()
        )
        return UPDATE_ACCOUNT

    elif user_choice == '🔍 Consultar Saldos':
        context.user_data['action_type'] = 'Consultar Saldos'
        await update.message.reply_text(
            "Elegí la cuenta que querés consultar:",
            reply_markup=make_account_keyboard()
        )
        return CONSULT_ACCOUNT

    elif user_choice in {'📋 Últimos movimientos', '📋 Últimos Movimientos'}:
        context.user_data['ultimos_origin'] = 'menu'
        await update.message.reply_text("Abramos tu historial ✨")
        return await ultimos_start(update, context)

    elif user_choice in {'📊 Resumen Capy', '📊 Resumen Capybara'}:
        await resumen_command(update, context)
        await update.message.reply_text(
            "¿Querés hacer otra cosa?",
            reply_markup=make_main_menu_keyboard()
        )
        return ACTION_TYPE

    elif user_choice in {'ℹ️ Ayuda', 'ℹ️ Ayuda Capy'}:
        await help_command(update, context)
        return ACTION_TYPE

    await update.message.reply_text(
        "No reconocí esa opción. Escribí /inicio para volver al menú.",
        reply_markup=make_main_menu_keyboard()
    )
    return ACTION_TYPE

# Monto
async def movement_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    context.user_data['movement_type'] = update.message.text
    logging.info("Tipo de movimiento de %s: %s", user.first_name, update.message.text)
    
    await update.message.reply_text(
        "Paso 1️⃣: decime el monto.\n💰 *Ejemplo:* 1000",
        parse_mode='Markdown'
    )
    return AMOUNT

# Moneda
async def amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Guardo Monto en la base
    user = update.message.from_user
    context.user_data['amount'] = update.message.text
    logging.info("Monto de %s: %s", user.first_name, update.message.text)
    
    # Pregunto por Moneda
    await update.message.reply_text(
        "Paso 2️⃣: elegí la *moneda*:",
        parse_mode='Markdown',
        reply_markup=ReplyKeyboardMarkup([['💲 Peso', '💵  Dólar', '👾 Cripto']], one_time_keyboard=True)
    )
    
    # Devuelvo variable Moneda
    return CURRENCY

# Descripcion
async def currency(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Guardo Moneda en la base
    user = update.message.from_user
    context.user_data['currency'] = update.message.text
    logging.info("Moneda de %s: %s", user.first_name, update.message.text)
    
    # Pregunto por Descripcion
    await update.message.reply_text(
        "Paso 3️⃣: elegí la **categoría** del movimiento:",
        parse_mode='Markdown',
        reply_markup=description_keyboard(context.user_data['movement_type'])
    )
    return DESCRIPTION

# Función para manejar la descripción
async def description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Guardo Descripcion en la base
    query = update.callback_query
    await query.answer()
    user = query.from_user
    context.user_data['description'] = query.data
    logging.info("Descripción de %s: %s", user.first_name, query.data)
    
    await query.message.reply_text(
        "¿De qué **forma** se realizó el movimiento?",
        parse_mode='Markdown',
        reply_markup=payment_method_keyboard()
    )
    return PAYMENT_METHOD

# Comentario
async def payment_method(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user = query.from_user
    context.user_data['payment_method'] = query.data
    logging.info("Forma de pago de %s: %s", user.first_name, query.data)

    await query.message.reply_text(
        "Paso 4️⃣: ¿querés agregar un **comentario**?",
        parse_mode='Markdown',
        reply_markup=make_yes_no_keyboard()
    )
    return COMMENT

# Función para manejar el comentario o continuar si es 'No'
async def comment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Verifico si el usuario quiere agregar un comentario
    if update.message.text == 'Sí':
        context.user_data['awaiting_comment'] = True  # Indicamos que estamos esperando un comentario
        await update.message.reply_text("Dale, escribí tu comentario ✍️")
        return CONTINUE
    else:
        # Si el usuario no quiere agregar un comentario, guardo un valor vacío
        context.user_data['comment'] = ""  # Comentario vacío si el usuario dijo "No"

        # Guardar la información en un CSV
        guardar_datos(update.effective_chat.id, context.user_data.copy())
        await update.message.reply_text(
            build_movement_confirmation(context.user_data),
            parse_mode='Markdown'
        )
        
        context.user_data['awaiting_comment'] = False
        limpiar_datos_movimiento(context)
        await update.message.reply_text(
            "🌟 Movimiento guardado. Elegí la próxima acción desde el menú 👇",
            reply_markup=make_main_menu_keyboard()
        )
        return ACTION_TYPE

# Función para manejar la respuesta del comentario
async def handle_comment_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Solo se ejecuta si estábamos esperando un comentario
    if context.user_data.get('awaiting_comment', False):
        # Guardo el comentario en la base de datos
        user = update.message.from_user
        context.user_data['comment'] = update.message.text
        logging.info("Comentario de %s: %s", user.first_name, update.message.text)
        
        # Guardar la información en un CSV
        guardar_datos(update.effective_chat.id, context.user_data.copy())
        await update.message.reply_text(
            build_movement_confirmation(context.user_data),
            parse_mode='Markdown'
        )
        
        context.user_data['awaiting_comment'] = False
        limpiar_datos_movimiento(context)
        await update.message.reply_text(
            "🌟 Movimiento guardado. Elegí la próxima acción desde el menú 👇",
            reply_markup=make_main_menu_keyboard()
        )
        return ACTION_TYPE

    # Si no se esperaba un comentario, manejamos la respuesta de continuar
    await update.message.reply_text(
        "No entendí ese comentario, pero podés elegir otra acción desde el menú 👇",
        reply_markup=make_main_menu_keyboard()
    )
    return ACTION_TYPE

# Función para manejar la cuenta elegida
async def update_account(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['account'] = update.message.text
    await update.message.reply_text(
        f"Escribí el nuevo saldo para *{update.message.text}* (solo número).",
        parse_mode='Markdown'
    )
    return UPDATE_AMOUNT

async def consult_account(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    account = update.message.text
    saldo = obtener_saldo(chat_id, account)
    if isinstance(saldo, str):
        await update.message.reply_text(saldo)
    else:
        await update.message.reply_text(
            f"🏦 *{account}*: {format_account_balance(account, saldo)}",
            parse_mode='Markdown'
        )
    context.user_data.pop('account', None)
    await update.message.reply_text(
        "Elegí la próxima acción desde el menú 👇",
        reply_markup=make_main_menu_keyboard()
    )
    return ACTION_TYPE

# Función para actualizar el saldo con el nuevo monto
async def update_amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    account = context.user_data['account']
    raw_amount = update.message.text.replace(',', '.')
    try:
        new_amount = float(raw_amount)
    except ValueError:
        await update.message.reply_text(
            "No pude entender el monto. Ingresá un número válido (ejemplo: 1234.56)."
        )
        return UPDATE_AMOUNT

    actualizar_saldo(chat_id, account, new_amount)
    await update.message.reply_text(
        f"✅ Saldo actualizado en *{account}*: {format_account_balance(account, new_amount)}",
        parse_mode='Markdown'
    )
    context.user_data.pop('account', None)
    await update.message.reply_text(
        "Elegí la próxima acción desde el menú 👇",
        reply_markup=make_main_menu_keyboard()
    )
    return ACTION_TYPE

# Función para consultar el saldo de una cuenta
async def consultar_saldo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    cuenta = update.message.text
    context.user_data['account'] = cuenta
    logging.info("Consulta de saldo por %s: %s", user.first_name, cuenta)
    
    # Obtener saldo de la cuenta
    saldo = obtener_saldo(update.effective_chat.id, cuenta)
    
    # Devolver el saldo al usuario
    if isinstance(saldo, str):
        await update.message.reply_text(saldo)
    else:
        await update.message.reply_text(
            f"🏦 *{cuenta}*: {format_account_balance(cuenta, saldo)}",
            parse_mode='Markdown'
        )
    
    # Preguntar si quiere hacer otra operación
    await update.message.reply_text(
        "Elegí la próxima acción desde el menú 👇",
        reply_markup=make_main_menu_keyboard()
    )
    return ACTION_TYPE

# Función para obtener el saldo de la cuenta desde saldos.csv
def obtener_saldo(chat_id: int, cuenta):
    df_saldos = load_saldos_dataframe(chat_id)
    if cuenta in df_saldos["cuenta"].values:
        saldo = df_saldos.loc[df_saldos["cuenta"] == cuenta, "saldo"].values[0]
        return saldo
    return "Cuenta no encontrada"

# Función para guardar datos en un CSV
def guardar_datos(chat_id: int, data: Dict[str, Any]):
    if DB_ENABLED:
        raw_amount = str(data.get('amount', '0')).replace(',', '.').strip()
        try:
            amount_value = float(raw_amount)
        except ValueError:
            amount_value = 0.0

        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(numero_movimiento), 0) + 1 FROM movements WHERE chat_id = %s",
                    (chat_id,)
                )
                next_number = cur.fetchone()[0]
                cur.execute(
                    """
                    INSERT INTO movements (
                        chat_id,
                        numero_movimiento,
                        movement_type,
                        amount,
                        currency,
                        description,
                        payment_method,
                        comment,
                        fecha
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        chat_id,
                        next_number,
                        data.get('movement_type'),
                        amount_value,
                        data.get('currency'),
                        data.get('description'),
                        data.get('payment_method'),
                        data.get('comment') or None,
                    ),
                )
            conn.commit()

        actualizar_saldo_por_movimiento(chat_id, data)
        return

    df = load_finanzas_dataframe(chat_id)

    new_data = pd.DataFrame([data])
    new_data["numero_movimiento"] = len(df) + 1
    new_data["fecha"] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data["chat_id"] = chat_id

    columns = [
        "chat_id",
        "numero_movimiento",
        "movement_type",
        "amount",
        "currency",
        "description",
        "payment_method",
        "comment",
        "fecha",
    ]
    new_data = new_data[columns]

    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(finanzas_path(chat_id), index=False)

    actualizar_saldo_por_movimiento(chat_id, data)


def limpiar_datos_movimiento(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Elimina datos temporales de movimiento para evitar contaminación entre flujos."""
    for key in (
        'action_type',
        'movement_type',
        'amount',
        'currency',
        'description',
        'payment_method',
        'comment',
        'awaiting_comment',
    ):
        context.user_data.pop(key, None)

# Función para actualizar el saldo basado en el movimiento registrado
def actualizar_saldo_por_movimiento(chat_id: int, data: Dict[str, Any]):
    metodo = data.get('payment_method')
    cuenta, ajustar = resolve_payment_method(metodo)

    if cuenta and ajustar:
        try:
            monto = float(data['amount'])
            if data['movement_type'] == '🚨 Gasto':
                monto = -monto

            actualizar_saldo(chat_id, cuenta, monto, modificar=True)
        except (TypeError, ValueError):
            logging.error('Error al convertir el monto del movimiento para actualizar saldo')

# Modificar la función de actualizar saldo para sumar o restar
def actualizar_saldo(chat_id: int, cuenta, monto, modificar=False):
    try:
        monto = float(monto)
    except (TypeError, ValueError):
        raise ValueError("Monto inválido para la actualización de saldo")

    if DB_ENABLED:
        with db_connection() as conn:
            with conn.cursor() as cur:
                if modificar:
                    cur.execute(
                        """
                        UPDATE balances
                           SET saldo = saldo + %s,
                               fecha_actualizacion = NOW()
                         WHERE chat_id = %s AND cuenta = %s
                        """,
                        (monto, chat_id, cuenta),
                    )
                    if cur.rowcount == 0:
                        cur.execute(
                            "INSERT INTO balances (chat_id, cuenta, saldo, fecha_actualizacion) VALUES (%s, %s, %s, NOW())",
                            (chat_id, cuenta, monto),
                        )
                else:
                    cur.execute(
                        """
                        INSERT INTO balances (chat_id, cuenta, saldo, fecha_actualizacion)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (chat_id, cuenta)
                        DO UPDATE SET saldo = EXCLUDED.saldo, fecha_actualizacion = EXCLUDED.fecha_actualizacion
                        """,
                        (chat_id, cuenta, monto),
                    )
            conn.commit()
        return

    df_saldos = load_saldos_dataframe(chat_id)

    if not df_saldos.empty:
        df_saldos["saldo"] = pd.to_numeric(df_saldos["saldo"], errors="coerce").fillna(0.0)

    fecha_actualizacion = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    if cuenta in df_saldos["cuenta"].values:
        if modificar:
            saldo_actual = df_saldos.loc[df_saldos["cuenta"] == cuenta, "saldo"].values[0]
            nuevo_saldo = float(saldo_actual) + monto
            df_saldos.loc[df_saldos["cuenta"] == cuenta, ["saldo", "fecha_actualizacion"]] = [nuevo_saldo, fecha_actualizacion]
        else:
            df_saldos.loc[df_saldos["cuenta"] == cuenta, ["saldo", "fecha_actualizacion"]] = [monto, fecha_actualizacion]
    else:
        new_row = pd.DataFrame([[chat_id, cuenta, monto, fecha_actualizacion]], columns=["chat_id", "cuenta", "saldo", "fecha_actualizacion"])
        df_saldos = pd.concat([df_saldos, new_row], ignore_index=True)

    if "chat_id" not in df_saldos.columns:
        df_saldos.insert(0, "chat_id", chat_id)
    else:
        df_saldos.loc[:, "chat_id"] = chat_id

    df_saldos.to_csv(saldos_path(chat_id), index=False)


async def add_account_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Uso: /add_account Nombre de la cuenta")
        return

    name = ' '.join(context.args).strip()
    if not name:
        await update.message.reply_text("El nombre de la cuenta no puede estar vacío.")
        return

    if find_account(name):
        await update.message.reply_text(f"La cuenta '{name}' ya existe.")
        return

    CONFIG.setdefault('accounts', []).append(name)
    save_config()
    refresh_config_cache()
    await update.message.reply_text(f"Cuenta '{name}' agregada correctamente.")


async def add_category_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args_text = ' '.join(context.args).strip()
    if not args_text:
        await update.message.reply_text(
            "Uso: /add_category tipo nombre [| identificador]. Ejemplo: /add_category gasto Cafetería | cafeteria"
        )
        return

    if '|' in args_text:
        main_part, slug_part = args_text.split('|', 1)
        slug_part = slug_part.strip()
    else:
        main_part, slug_part = args_text, None

    tokens = main_part.split()
    if len(tokens) < 2:
        await update.message.reply_text(
            "Debes indicar el tipo (ingreso/gasto) y el nombre de la categoría."
        )
        return

    category_type = tokens[0].lower()
    category_map = {
        'ingreso': 'income',
        'income': 'income',
        'gasto': 'expense',
        'expense': 'expense',
    }
    category_key = category_map.get(category_type)
    if not category_key:
        await update.message.reply_text("Tipo de categoría inválido. Usa ingreso o gasto.")
        return

    label = ' '.join(tokens[1:]).strip()
    if not label:
        await update.message.reply_text("El nombre de la categoría no puede estar vacío.")
        return

    target_list = CATEGORIES[category_key]
    if any(item['text'].lower() == label.lower() for item in target_list):
        await update.message.reply_text(f"La categoría '{label}' ya existe.")
        return

    base_value = slug_part if slug_part else slugify(label)
    candidate = base_value
    counter = 1
    existing_values = {item['value'].lower() for item in target_list}
    while candidate.lower() in existing_values:
        candidate = f"{base_value}_{counter}"
        counter += 1

    target_list.append({'text': label, 'value': candidate})
    save_config()
    refresh_config_cache()
    await update.message.reply_text(
        f"Categoría '{label}' agregada al listado de {category_key}. Identificador: {candidate}"
    )


async def add_payment_method_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args_text = ' '.join(context.args).strip()
    if not args_text:
        await update.message.reply_text(
            "Uso: /add_payment_method Nombre [| Cuenta vinculada] [| ajusta_saldo]. Ejemplo: /add_payment_method Transferencia UALA | UALA"
        )
        return

    parts = [part.strip() for part in args_text.split('|')]
    name = parts[0]
    if not name:
        await update.message.reply_text("El nombre del método de pago no puede estar vacío.")
        return

    if payment_method_exists(name):
        await update.message.reply_text(f"El método de pago '{name}' ya existe.")
        return

    account = parts[1] if len(parts) > 1 and parts[1] else None
    explicit_adjust = len(parts) > 2 and parts[2] != ''
    adjust_balance = True
    if explicit_adjust:
        adjust_balance = parts[2].lower() in {'true', '1', 'si', 'sí', 'yes', 'y'}

    added_account = None
    if account:
        existing_account = find_account(account)
        if not existing_account:
            CONFIG.setdefault('accounts', []).append(account)
            added_account = account
        else:
            account = existing_account
    else:
        inferred_account = infer_account_from_name(name)
        account = inferred_account

    entry = {'name': name}
    if account:
        entry['account'] = account
    if explicit_adjust:
        entry['adjust_balance'] = bool(adjust_balance)

    CONFIG.setdefault('payment_methods', []).append(entry)
    save_config()
    refresh_config_cache()

    messages = [f"Método de pago '{name}' agregado."]
    if account:
        messages.append(f"Se vincula con la cuenta '{account}'.")
    if explicit_adjust and not adjust_balance:
        messages.append("No ajustará saldos automáticamente.")
    if added_account:
        messages.append(f"Se creó también la cuenta '{added_account}'.")

    await update.message.reply_text(' '.join(messages))


async def ultimos_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if 'ultimos_origin' not in context.user_data:
        context.user_data['ultimos_origin'] = 'command'

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton('🔟 Últimos 10', callback_data='ULTIMOS_RECENT')],
        [InlineKeyboardButton('🎯 Filtro avanzado', callback_data='ULTIMOS_FILTER')],
    ])

    await update.message.reply_text(
        "¿Qué movimientos querés consultar?",
        reply_markup=keyboard
    )
    return ULTIMOS_CHOICE


async def ultimos_handle_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if not query:
        return ConversationHandler.END

    await query.answer()
    choice = query.data
    origin = context.user_data.get('ultimos_origin', 'command')

    if choice == 'ULTIMOS_RECENT':
        df = load_finanzas_dataframe(query.message.chat_id).sort_values(by='fecha', ascending=False)
        message = build_movements_message(df, limit=10)
        await query.message.reply_text(message)
        context.user_data.pop('ultimos_origin', None)
        if origin == 'menu':
            await query.message.reply_text(
                "Elegí la próxima acción en el menú 👇",
                reply_markup=make_main_menu_keyboard()
            )
            return ACTION_TYPE
        await query.message.reply_text(
            "Escribí /inicio para volver al menú o usá /movimientos nuevamente.")
        return ConversationHandler.END

    if choice == 'ULTIMOS_FILTER':
        context.user_data['awaiting_last_filter'] = True
        instructions = (
            "Ingresá los filtros separados por ';'. Ejemplo:\n"
            "cuenta=Mercado Pago; desde=2024-01-01; hasta=2024-01-31; tipo=gasto; categoria=Supermercado; metodo=Transferencia; comentario=almuerzo"
        )
        await query.message.reply_text(instructions)
        return ULTIMOS_FILTER

    return ConversationHandler.END


async def ultimos_handle_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not context.user_data.get('awaiting_last_filter'):
        origin = context.user_data.pop('ultimos_origin', 'command')
        if origin == 'menu':
            return ACTION_TYPE
        return ConversationHandler.END

    filters = parse_filter_text(update.message.text)
    chat_id = update.effective_chat.id
    df = load_finanzas_dataframe(chat_id)
    if df.empty:
        await update.message.reply_text('No tenés movimientos registrados aún.')
        context.user_data.pop('awaiting_last_filter', None)
        origin = context.user_data.pop('ultimos_origin', 'command')
        if origin == 'menu':
            await update.message.reply_text(
                "Elegí la próxima acción en el menú 👇",
                reply_markup=make_main_menu_keyboard()
            )
            return ACTION_TYPE
        return ConversationHandler.END

    filtered = filter_movements(df, filters)
    message = build_movements_message(filtered, limit=20)
    await update.message.reply_text(message)
    origin = context.user_data.pop('ultimos_origin', 'command')
    context.user_data.pop('awaiting_last_filter', None)
    if origin == 'menu':
        await update.message.reply_text(
            "Elegí la próxima acción en el menú 👇",
            reply_markup=make_main_menu_keyboard()
        )
        return ACTION_TYPE

    await update.message.reply_text("Escribí /inicio para volver al menú o seguí consultando.")
    return ConversationHandler.END


async def resumen_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    df_finanzas = load_finanzas_dataframe(chat_id)
    df_saldos = load_saldos_dataframe(chat_id)

    totals = compute_account_totals(df_saldos)
    month_totals = compute_monthly_totals(df_finanzas)
    summary_text = build_summary_message(totals, month_totals)

    await update.message.reply_text(summary_text, parse_mode='Markdown')

    chart_buffer = generate_expense_chart(df_finanzas)
    if chart_buffer:
        chart_buffer.name = 'gastos_mes.png'
        await update.message.reply_photo(InputFile(chart_buffer, filename='gastos_mes.png'))
    else:
        has_expenses = not df_finanzas[df_finanzas['movement_type'] == '🚨 Gasto'].empty
        if has_expenses:
            await update.message.reply_text(
                "No pude generar gráfico de gastos (quizás no hay gastos este mes o falta matplotlib)."
            )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            HELP_MESSAGE,
            parse_mode='Markdown',
            reply_markup=make_main_menu_keyboard()
        )
    elif update.callback_query:
        query = update.callback_query
        await query.answer()
        await query.message.reply_text(
            HELP_MESSAGE,
            parse_mode='Markdown',
            reply_markup=make_main_menu_keyboard()
        )


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text('Operación cancelada. ¡Hasta luego!')
    return ConversationHandler.END


async def post_init(application: Application) -> None:
    try:
        await application.bot.set_my_commands(BOT_COMMANDS_CONFIG)
    except Exception as exc:
        logging.warning('No se pudieron registrar los comandos del bot: %s', exc)


def main() -> None:
    init_db()
    application = Application.builder().token(TOKEN).post_init(post_init).build()
    
    application.add_handler(CommandHandler('ayuda', help_command))
    application.add_handler(CommandHandler('agregar_cuenta', add_account_command))
    application.add_handler(CommandHandler('agregar_categoria', add_category_command))
    application.add_handler(CommandHandler('agregar_medio_pago', add_payment_method_command))
    application.add_handler(CommandHandler('resumen', resumen_command))

    conv_handler = ConversationHandler(
    entry_points=[CommandHandler('inicio', start)],
    states={
        ACTION_TYPE: [MessageHandler(
            filters.Regex('^(➕ Registrar movimiento|📋 Últimos movimientos|💰 Ajustar saldos|🔍 Consultar saldos|📊 Resumen Capy|ℹ️ Ayuda)$'),
            action_type
        )],
        MOVEMENT_TYPE: [MessageHandler(filters.Regex('^(🤑 Ingreso|🚨 Gasto)$'), movement_type)],
        UPDATE_ACCOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, update_account)],
        CONSULT_ACCOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, consult_account)],
        UPDATE_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, update_amount)],
        AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, amount)],
        CURRENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND, currency)],
        COMMENT: [MessageHandler(filters.Regex('^(Sí|No)$'), comment)],
        CONTINUE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment_text),
        ],
        DESCRIPTION: [CallbackQueryHandler(description)],
        PAYMENT_METHOD: [CallbackQueryHandler(payment_method)],
        ULTIMOS_CHOICE: [CallbackQueryHandler(ultimos_handle_choice, pattern='^ULTIMOS_')],
        ULTIMOS_FILTER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ultimos_handle_filter)],
    },
    fallbacks=[CommandHandler('cancel', cancel)],
    allow_reentry=True,
)

    application.add_handler(conv_handler)

    ultimos_handler = ConversationHandler(
        entry_points=[CommandHandler('ultimos', ultimos_start)],
        states={
            ULTIMOS_CHOICE: [CallbackQueryHandler(ultimos_handle_choice, pattern='^ULTIMOS_')],
            ULTIMOS_FILTER: [MessageHandler(filters.TEXT & ~filters.COMMAND, ultimos_handle_filter)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
        allow_reentry=True,
    )

    application.add_handler(ultimos_handler)

    application.run_polling()

if __name__ == '__main__':
    main()
