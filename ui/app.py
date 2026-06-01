"""Interfaz web Gradio del Asistente Normativo de Unillanos.

Ejecutar:
    python ui/app.py
    python ui/app.py --port 7860 --share
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gradio as gr
from src.factory import create_pipeline

_pipeline = None


def _get_pipeline(config_path: str = "config/config.yaml"):
    global _pipeline
    if _pipeline is None:
        _pipeline = create_pipeline(config_path)
    return _pipeline


def _respond(message: str, history: list[list]) -> tuple[str, list[list]]:
    if not message.strip():
        return "", history
    pipeline = _get_pipeline()
    result = pipeline.query(message)
    history.append([message, result["answer"]])
    return "", history


def _clear_history() -> list:
    _get_pipeline().reset_history()
    return []


def build_app(config_path: str = "config/config.yaml") -> gr.Blocks:
    pipeline = _get_pipeline(config_path)
    model_info = pipeline.model_name

    with gr.Blocks(
        title="Asistente Normativo Unillanos",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=".gradio-container { max-width: 900px; margin: auto; }",
    ) as demo:

        gr.Markdown(f"""# Asistente Normativo — Universidad de los Llanos

Consulta sobre **resoluciones, acuerdos y normativa universitaria**.

> **Modelo activo:** `{model_info}`
""")

        chatbot = gr.Chatbot(
            label="Conversación",
            height=480,
            bubble_full_width=False,
            show_label=False,
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Escribe tu pregunta sobre normativa universitaria...",
                show_label=False,
                scale=8,
                lines=1,
            )
            send_btn = gr.Button("Enviar", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Nueva conversación", variant="secondary", scale=1)

        gr.Examples(
            examples=[
                ["¿Cuándo es el calendario de grados para 2026?"],
                ["¿Qué dice la resolución sobre fraccionamiento de matrícula?"],
                ["¿Cuáles son las opciones de grado de la Facultad de Ciencias Básicas?"],
                ["¿Qué establece la resolución sobre apoyo económico a docentes catedráticos?"],
                ["¿Qué resuelve la resolución académica 043 de 2026?"],
            ],
            inputs=msg_box,
            label="Preguntas de ejemplo",
        )

        msg_box.submit(_respond, [msg_box, chatbot], [msg_box, chatbot])
        send_btn.click(_respond, [msg_box, chatbot], [msg_box, chatbot])
        clear_btn.click(_clear_history, [], chatbot)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Exponer via Gradio tunnel")
    args = parser.parse_args()

    print("=" * 60)
    print("  Asistente Normativo Unillanos — Interfaz Web")
    print("=" * 60)
    print("  Cargando pipeline...")

    app_instance = build_app(args.config)

    print(f"  Iniciando en http://{args.host}:{args.port}")
    print("  Ctrl+C para detener\n")

    app_instance.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
