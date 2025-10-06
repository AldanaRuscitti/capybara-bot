"""Script para migrar datos desde CSV locales a la base de datos Postgres."""

import os
from pathlib import Path

import pandas as pd
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL no está definido. Configurá la variable de entorno con la cadena de conexión de Railway."
    )

BOT_DATA_DIR = os.environ.get("BOT_DATA_DIR", ".")


def migrate_user(chat_id: str) -> None:
    """Migra los archivos CSV de un chat específico a Postgres."""
    data_dir = Path(BOT_DATA_DIR) / chat_id
    finanzas_file = data_dir / "finanzas.csv"
    saldos_file = data_dir / "saldos.csv"

    if not finanzas_file.exists() and not saldos_file.exists():
        print(f"No se encontraron CSV para el chat {chat_id} en {data_dir}.")
        return

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            if finanzas_file.exists():
                df_mov = pd.read_csv(finanzas_file)
                for _, row in df_mov.iterrows():
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
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            row.get("chat_id", chat_id),
                            row.get("numero_movimiento", 0),
                            row.get("movement_type"),
                            row.get("amount"),
                            row.get("currency"),
                            row.get("description"),
                            row.get("payment_method"),
                            row.get("comment"),
                            row.get("fecha"),
                        ),
                    )

            if saldos_file.exists():
                df_bal = pd.read_csv(saldos_file)
                for _, row in df_bal.iterrows():
                    cur.execute(
                        """
                        INSERT INTO balances (chat_id, cuenta, saldo, fecha_actualizacion)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (chat_id, cuenta)
                        DO UPDATE SET saldo = EXCLUDED.saldo,
                                      fecha_actualizacion = EXCLUDED.fecha_actualizacion
                        """,
                        (
                            row.get("chat_id", chat_id),
                            row.get("cuenta"),
                            row.get("saldo"),
                            row.get("fecha_actualizacion"),
                        ),
                    )
        conn.commit()

    print(f"Migración completada para el chat {chat_id}.")


def migrate_all_chats() -> None:
    """Recorre el directorio BOT_DATA_DIR e intenta migrar cada chat."""
    base = Path(BOT_DATA_DIR)
    if not base.exists():
        raise RuntimeError(f"El directorio {BOT_DATA_DIR} no existe.")

    for child in base.iterdir():
        if child.is_dir() and child.name.isdigit():
            migrate_user(child.name)


if __name__ == "__main__":
    # Si querés migrar todos los chats del directorio, ejecutá el script sin cambios.
    migrate_all_chats()
    # Para migrar uno en particular, comentá la línea anterior y usá:
    # migrate_user("123456789")
