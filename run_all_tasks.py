#!/usr/bin/env python3
"""
Run All Tasks - Complete AI/Data Engineering Assignment
Orchestrates execution of all three tasks in sequence
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskRunner:
    """Orchestrate execution of all assignment tasks"""
    
    def __init__(self):
        self.processes = []
        self.task_results = {}
    
    def setup_environment(self):
        """Setup environment and validate configuration"""
        print("🛠️ Setting up environment...")
        
        # Validate configuration
        Config.validate_config()
        Config.ensure_directories()
        
        # Check required files
        sample_dir = Path(Config.SAMPLE_IMAGES_DIR)
        if not any(sample_dir.iterdir()) if sample_dir.exists() else True:
            print("⚠️  No sample images found!")
            print(f"   Please add some document images to: {sample_dir}")
            print("   Supported formats: .jpg, .jpeg, .png, .pdf, .tiff, .bmp")
            
            # Create a sample file for testing
            sample_file = sample_dir / "README.txt"
            with open(sample_file, 'w') as f:
                f.write("Sample document for testing AI/Data Engineering Assignment")
            print(f"   Created sample file: {sample_file}")
        
        print("✅ Environment setup complete")
    
    def run_task1(self):
        """Execute Task 1: OCR and NLP Processing"""
        print("\n" + "="*60)
        print("🚀 TASK 1: OCR Implementation and NLP Processing")
        print("="*60)
        
        try:
            # Import and run task 1
            from src.task1_ocr_nlp import main as task1_main
            
            start_time = time.time()
            task1_main()
            execution_time = time.time() - start_time
            
            self.task_results['task1'] = {
                'status': 'completed',
                'execution_time': execution_time,
                'description': 'OCR comparison and NLP feature extraction'
            }
            
            print(f"✅ Task 1 completed in {execution_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Task 1 failed: {e}")
            self.task_results['task1'] = {
                'status': 'failed',
                'error': str(e),
                'description': 'OCR comparison and NLP feature extraction'
            }
    
    def run_task2_server(self):
        """Start Task 2: FastAPI Server"""
        print("\n" + "="*60)
        print("🚀 TASK 2: LLM Summarization and Database Storage")
        print("="*60)
        
        try:
            # Start FastAPI server in background
            cmd = [
                sys.executable, "-m", "uvicorn",
                "src.task2_llm_storage:app",
                "--host", Config.API_HOST,
                "--port", str(Config.API_PORT),
                "--reload"
            ]
            
            print(f"🌐 Starting API server on http://{Config.API_HOST}:{Config.API_PORT}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(('task2_api', process))
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Test server health
            import requests
            try:
                response = requests.get(f"http://localhost:{Config.API_PORT}/health", timeout=5)
                if response.status_code == 200:
                    print("✅ API server is running and healthy")
                    self.task_results['task2_api'] = {
                        'status': 'running',
                        'description': 'FastAPI server for document processing',
                        'url': f"http://localhost:{Config.API_PORT}"
                    }
                else:
                    print(f"⚠️ API server responded with status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Could not connect to API server: {e}")
            
        except Exception as e:
            logger.error(f"Failed to start Task 2 API server: {e}")
            self.task_results['task2_api'] = {
                'status': 'failed',
                'error': str(e),
                'description': 'FastAPI server for document processing'
            }
    
    def run_task3_chatbot(self):
        """Start Task 3: Chainlit Chatbot"""
        print("\n" + "="*60)
        print("🚀 TASK 3: AI Chatbot with Chainlit Frontend")
        print("="*60)
        
        try:
            # Start Chainlit chatbot in background
            cmd = [
                "chainlit", "run", "src/task3_chatbot.py",
                "-w", "--port", str(Config.CHAINLIT_PORT)
            ]
            
            print(f"💬 Starting chatbot on http://localhost:{Config.CHAINLIT_PORT}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(('task3_chatbot', process))
            
            # Wait a moment for chatbot to start
            time.sleep(3)
            
            print("✅ Chainlit chatbot is starting...")
            self.task_results['task3_chatbot'] = {
                'status': 'running',
                'description': 'Chainlit conversational interface',
                'url': f"http://localhost:{Config.CHAINLIT_PORT}"
            }
            
        except Exception as e:
            logger.error(f"Failed to start Task 3 chatbot: {e}")
            self.task_results['task3_chatbot'] = {
                'status': 'failed',
                'error': str(e),
                'description': 'Chainlit conversational interface'
            }
    
    def test_system(self):
        """Test the complete system integration"""
        print("\n" + "="*60)
        print("🧪 SYSTEM INTEGRATION TESTING")
        print("="*60)
        
        try:
            import requests
            
            base_url = f"http://localhost:{Config.API_PORT}"
            
            # Test 1: Health check
            print("1. Testing API health...")
            try:
                response = requests.get(f"{base_url}/health", timeout=5)
                if response.status_code == 200:
                    print("   ✅ API health check passed")
                else:
                    print(f"   ❌ API health check failed: {response.status_code}")
            except Exception as e:
                print(f"   ❌ API health check failed: {e}")
            
            # Test 2: Document upload (if sample files exist)
            sample_dir = Path(Config.SAMPLE_IMAGES_DIR)
            sample_files = list(sample_dir.glob("*"))
            
            if sample_files:
                print("2. Testing document upload...")
                test_file = sample_files[0]
                
                try:
                    with open(test_file, 'rb') as f:
                        files = {'file': (test_file.name, f, 'image/jpeg')}
                        response = requests.post(f"{base_url}/upload", files=files, timeout=10)
                    
                    if response.status_code == 200:
                        print(f"   ✅ Document upload successful: {test_file.name}")
                    else:
                        print(f"   ❌ Document upload failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ Document upload failed: {e}")
            else:
                print("2. Skipping upload test - no sample files found")
            
            # Test 3: Search functionality
            print("3. Testing search functionality...")
            try:
                search_data = {"query": "test document", "top_k": 3}
                response = requests.post(
                    f"{base_url}/search", 
                    json=search_data, 
                    timeout=10
                )
                
                if response.status_code == 200:
                    results = response.json()
                    print(f"   ✅ Search successful: found {results.get('total_found', 0)} results")
                else:
                    print(f"   ❌ Search failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Search test failed: {e}")
            
            print("\n✔️ System testing completed")
            
        except Exception as e:
            logger.error(f"System testing failed: {e}")
    
    def display_results(self):
        """Display final results and access information"""
        print("\n" + "="*60)
        print("📋 ASSIGNMENT COMPLETION SUMMARY")
        print("="*60)
        
        for task_name, result in self.task_results.items():
            status = result['status']
            description = result['description']
            
            if status == 'completed':
                time_str = f" ({result['execution_time']:.2f}s)" if 'execution_time' in result else ""
                print(f"✅ {task_name}: {description}{time_str}")
            elif status == 'running':
                url = result.get('url', '')
                print(f"⚡ {task_name}: {description}")
                if url:
                    print(f"   🌐 Access at: {url}")
            elif status == 'failed':
                print(f"❌ {task_name}: {description}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print("\n🔗 Access Points:")
        if 'task2_api' in self.task_results and self.task_results['task2_api']['status'] == 'running':
            print(f"   📖 API Documentation: http://localhost:{Config.API_PORT}/docs")
            print(f"   🩺 Health Check: http://localhost:{Config.API_PORT}/health")
        
        if 'task3_chatbot' in self.task_results and self.task_results['task3_chatbot']['status'] == 'running':
            print(f"   💬 Chatbot Interface: http://localhost:{Config.CHAINLIT_PORT}")
        
        print("\n⚡ Quick Test Commands:")
        print(f"   curl -X GET 'http://localhost:{Config.API_PORT}/health'")
        print(f"   curl -X POST 'http://localhost:{Config.API_PORT}/upload' -F 'file=@sample.jpg'")
        
        print("\n🛑 To stop all services:")
        print("   Press Ctrl+C in this terminal")
    
    def cleanup(self):
        """Cleanup processes on exit"""
        print("\n🛑 Stopping all services...")
        
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"   ✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"   ⚠️ Force killed {name}")
            except Exception as e:
                print(f"   ⚠️ Error stopping {name}: {e}")
    
    def run_all(self):
        """Run all tasks in sequence"""
        try:
            # Setup
            self.setup_environment()
            
            # Task 1: OCR and NLP (runs to completion)
            self.run_task1()
            
            # Task 2: API Server (runs in background)
            self.run_task2_server()
            
            # Task 3: Chatbot (runs in background)  
            self.run_task3_chatbot()
            
            # Wait for services to stabilize
            time.sleep(5)
            
            # Test system integration
            self.test_system()
            
            # Display results
            self.display_results()
            
            # Keep services running
            print("\n⏳ Services are running. Press Ctrl+C to stop all services.")
            
            # Wait for interrupt
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 Received interrupt signal...")
                
        except Exception as e:
            logger.error(f"Failed to run tasks: {e}")
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    print("🚀 AI/Data Engineering Assignment - Complete Implementation")
    print("=" * 60)
    print("This script will run all three tasks:")
    print("1. OCR Implementation and NLP Processing")
    print("2. LLM Summarization and Database Storage")
    print("3. AI Chatbot with Chainlit Frontend")
    print("=" * 60)
    
    # Signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Received shutdown signal")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run all tasks
    runner = TaskRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
