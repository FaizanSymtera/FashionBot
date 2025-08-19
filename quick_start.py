#!/usr/bin/env python3
"""
Quick Start Script for Fashion Chatbot Backend
This script helps you set up and run the MongoDB-integrated backend.
"""

import os
import sys
import subprocess
import time

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # First, try to upgrade pip
        print("🔄 Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Try installing dependencies
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        
        # If pandas fails, try installing it separately with a compatible version
        print("\n🔄 Trying alternative installation method...")
        try:
            # Install packages one by one, starting with pandas
            print("📦 Installing pandas with compatible version...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas>=2.2.0"])
            
            print("📦 Installing other dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Flask>=2.3.0", "Flask-CORS>=4.0.0", "python-dotenv>=1.0.0", "google-generativeai>=0.3.0", "pymongo>=4.5.0", "requests>=2.31.0"])
            
            print("✅ Dependencies installed successfully with alternative method!")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Alternative installation also failed: {e2}")
            print("\n💡 Manual installation suggestions:")
            print("1. Try installing pandas separately: pip install pandas>=2.2.0")
            print("2. If that fails, try: pip install pandas --no-build-isolation")
            print("3. Or use conda: conda install pandas")
            print("4. Consider using Python 3.11 or 3.12 if pandas installation continues to fail")
            return False

def check_mongodb():
    """Check if MongoDB is accessible."""
    print("\n🔍 Checking MongoDB connection...")
    
    try:
        import pymongo
        from pymongo import MongoClient
        
        # Try to connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ MongoDB is accessible!")
        client.close()
        return True
        
    except ImportError:
        print("❌ PyMongo not installed. Installing now...")
        return install_dependencies()
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n💡 Please ensure MongoDB is running:")
        print("   - Install MongoDB: https://docs.mongodb.com/manual/installation/")
        print("   - Or use MongoDB Atlas: https://www.mongodb.com/atlas")
        print("   - Update MONGODB_URI in config.json if using a different connection string")
        return False

def setup_database():
    """Set up the MongoDB database."""
    print("\n🗄️  Setting up MongoDB database...")
    
    try:
        subprocess.check_call([sys.executable, "setup_mongodb.py"])
        print("✅ Database setup completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Database setup failed.")
        return False

def check_config():
    """Check if configuration is properly set up."""
    print("\n⚙️  Checking configuration...")
    
    try:
        with open('config.json', 'r') as f:
            config = eval(f.read())
        
        required_keys = ['GOOGLE_API_KEY', 'EXCHANGE_RATE_API_KEY', 'MONGODB_URI']
        missing_keys = [key for key in required_keys if key not in config or not config[key]]
        
        if missing_keys:
            print(f"❌ Missing or empty configuration keys: {', '.join(missing_keys)}")
            print("💡 Please update config.json with your API keys")
            return False
        
        print("✅ Configuration looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False

def start_backend():
    """Start the backend server."""
    print("\n🚀 Starting backend server...")
    print("💡 The server will run on http://localhost:5000")
    print("💡 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, "FinetunnedBackendCode.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user.")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main function."""
    print("🎉 Welcome to Fashion Chatbot Backend Setup!")
    print("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Step 3: Check MongoDB
    if not check_mongodb():
        sys.exit(1)
    
    # Step 4: Setup database
    if not setup_database():
        sys.exit(1)
    
    # Step 5: Check configuration
    if not check_config():
        print("\n💡 Please update your configuration and run this script again.")
        sys.exit(1)
    
    # Step 6: Start backend
    print("\n🎯 All checks passed! Starting the backend server...")
    start_backend()

if __name__ == "__main__":
    main()
