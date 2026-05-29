from src.model_trainer import train_model
from src.logger import console
from rich.panel import Panel

def run_training_pipeline():
    """
    Executes the training pipeline with professional CLI output.
    """
    console.print(Panel.fit(
        "Patient Journey & Churn Analysis\n"
        "Advanced Analytics Pipeline",
        border_style="blue"
    ))

    try:
        results = train_model(save=True)
        console.print(f"\n[bold green]Success:[/bold green] Model trained with AUC-ROC: [bold cyan]{results['auc']:.4f}[/bold cyan]")
        console.print(f"Deployment command: streamlit run app.py\n")
        
    except Exception as e:
        console.print(f"\n[bold red]Error: Pipeline execution failed:[/bold red] {str(e)}")

if __name__ == "__main__":
    run_training_pipeline()
