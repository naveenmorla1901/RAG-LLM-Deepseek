import asyncio
import argparse
import sys
from pathlib import Path
import logging
from typing import Optional, List, Dict, AsyncGenerator
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table
from rag_chain import LoanAssistant
from config import validate_environment, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

def format_response(response: Dict):
    """Format the response in a clear, structured way"""
    console.print("\n" + "-" * 50)
    
    # Show response source and status header
    if response.get("is_backup_response", False):
        console.print("[bold red]⚠ SYSTEM BACKUP RESPONSE (Due to Connection Issues)[/bold red]")
        console.print("[bold red]Using built-in fallback knowledge[/bold red]")
    elif response.get("is_general_answer", False):
        console.print("[bold yellow]ℹ MODEL KNOWLEDGE RESPONSE[/bold yellow]")
        console.print("[bold yellow]No relevant documents found - Using model's base knowledge[/bold yellow]")
    else:
        console.print("[bold green]📄 DOCUMENT-BASED RESPONSE[/bold green]")
        console.print("[bold green]Answer derived from analyzed documents[/bold green]")
    
    console.print("\n[bold]Answer:[/bold]")
    console.print(response["answer"].strip())
    
    # Print references if they exist
    if response["references"]:
        console.print("\n[bold blue]References Used:[/bold blue]")
        console.print("-" * 50)
        
        for ref in response["references"]:
            console.print(f"→ {ref['source']}, {ref['location']}")
            console.print(f"  Content: {ref['content']}")
            console.print(f"  ID: {ref['id']}")
            console.print()
    
    # Print source summary
    source_type = (
        "🔄 Source: System Backup Knowledge" if response.get("is_backup_response", False)
        else "🧠 Source: Model's Base Knowledge" if response.get("is_general_answer", False)
        else "📚 Source: Local Document Analysis"
    )
    console.print(f"\n[bold]{source_type}[/bold]")
    
    console.print("-" * 50 + "\n")

async def interactive_mode(assistant: LoanAssistant):
    """Run interactive query mode"""
    console.print(Panel(
        "[bold]Welcome to the Loan Document Analysis Assistant[/bold]\n\n"
        "Type your questions about loan documents, applications, or policies.\n"
        "Use /help to see available commands.",
        border_style="blue"
    ))

    while True:
        try:
            # Get user input
            question = Prompt.ask("\n[bold blue]Ask a question[/bold blue]")
            
            # Handle commands
            if question.lower() == '/exit':
                break
            elif question.lower() == '/help':
                display_help()
                continue
            elif question.lower() == '/files':
                await list_processed_files(assistant)
                continue
            elif question.lower() == '/clear':
                console.clear()
                continue
            elif question.lower().startswith('/ingest '):
                file_path = question[8:].strip()
                await ingest_new_document(assistant, file_path)
                continue
            elif question.lower().startswith('/analyze '):
                app_id = question[9:].strip()
                await analyze_application(assistant, app_id)
                continue

            # Process regular questions
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                response = await assistant.answer_question(question, streaming=False)
                progress.update(task, completed=True)

            # Format and display response
            format_response(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Use /exit to quit the program[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

def display_help():
    """Display available commands and usage information"""
    help_text = """
    Available Commands:
    ------------------
    /help           - Show this help message
    /analyze <id>   - Analyze a specific loan application
    /files          - List processed documents
    /ingest <path>  - Process a new document
    /clear         - Clear the screen
    /exit          - Exit the program
    
    Tips:
    -----
    - Ask questions about loan requirements, policies, or specific applications
    - Use specific application IDs when asking about particular cases
    - Request document analysis for completeness checks
    """
    console.print(Panel(help_text, title="Help Information", border_style="blue"))

async def list_processed_files(assistant: LoanAssistant):
    """Display list of processed documents"""
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("File Name")
    table.add_column("Type")
    table.add_column("Chunks")
    table.add_column("Last Updated")

    for file_info in assistant.doc_processor.processed_files.items():
        file_path = Path(file_info[0])
        table.add_row(
            file_path.name,
            file_path.suffix[1:].upper(),
            str(len(assistant.collection.get(
                where={"source": str(file_path)}
            )['ids'])),
            "Recently"  # TODO: Add actual timestamp
        )

    console.print(Panel(table, title="Processed Documents", border_style="blue"))

async def ingest_new_document(assistant: LoanAssistant, file_path: str):
    """Process a new document with progress feedback"""
    try:
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {path.name}...", total=None)
            result = await assistant.ingest_document(path)
            progress.update(task, completed=True)

        if result["status"] == "success":
            console.print(f"[green]✓ Successfully processed {path.name}[/green]")
        else:
            console.print(f"[red]Error processing {path.name}: {result.get('error', 'Unknown error')}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

async def analyze_application(assistant: LoanAssistant, application_id: str):
    """Analyze a specific loan application"""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Analyzing application {application_id}...", total=None)
            result = await assistant.analyze_loan_application(application_id)
            progress.update(task, completed=True)

        if result["status"] == "success":
            console.print(Panel(
                Markdown(result["analysis"]),
                title=f"Analysis for Application {application_id}",
                border_style="green"
            ))
        else:
            console.print(f"[red]Error analyzing application: {result.get('error', 'Unknown error')}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

async def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Loan Document Analysis System")
        parser.add_argument('--model', default='deepseek-chat',
                          choices=['deepseek-chat', 'gpt-4', 'claude-3'],
                          help='LLM model to use')
        args = parser.parse_args()

        # Validate environment
        validate_environment()

        # Initialize assistant
        assistant = LoanAssistant(model_name=args.model)

        # Run interactive mode
        await interactive_mode(assistant)

    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)
def format_response(response: Dict):
    """Format the response in a clear, structured way"""
    console.print("\n" + "-" * 50)
    
    # Print if this is a general answer or document-based answer
    if response.get("is_backup_response", False):
        console.print("[bold red]System Notice: Using Backup Response (Connection Issues)[/bold red]")
    elif response.get("is_general_answer", False):
        console.print("[bold yellow]General Information (Not from loaded documents):[/bold yellow]")
    else:
        console.print("[bold green]Answer from Documents:[/bold green]")
    
    console.print(response["answer"].strip())
    
    # Print references if they exist
    if response["references"]:
        console.print("\n[bold blue]References Used:[/bold blue]")
        console.print("-" * 50)
        
        for ref in response["references"]:
            console.print(f"→ {ref['source']}, {ref['location']}")
            console.print(f"  Content: {ref['content']}")
            console.print(f"  ID: {ref['id']}")
            console.print()
    
    console.print("-" * 50 + "\n")
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Program terminated by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        sys.exit(1)