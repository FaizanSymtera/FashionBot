#!/usr/bin/env python3
"""
MongoDB Setup Script for Fashion Chatbot
This script helps set up and manage the MongoDB database for the fashion chatbot.
"""

import os
import sys
import json
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import datetime

def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    try:
        # Try to get MongoDB URI from environment or config
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        database_name = os.getenv('MONGODB_DATABASE', 'fashion_chatbot')
        
        # Try to read from config.json if environment variables not set
        if mongo_uri == 'mongodb://localhost:27017/':
            try:
                with open('config.json', 'r') as f:
                    config = json.load(f)
                mongo_uri = config.get('MONGODB_URI', 'mongodb://localhost:27017/')
                database_name = config.get('MONGODB_DATABASE', 'fashion_chatbot')
            except FileNotFoundError:
                print("⚠️  config.json not found, using default MongoDB settings")
        
        print(f"🔗 Connecting to MongoDB at: {mongo_uri}")
        print(f"📁 Database name: {database_name}")
        
        # Connect with timeout
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
        
        db = client[database_name]
        return db, client
        
    except ConnectionFailure:
        print("❌ Failed to connect to MongoDB. Please ensure MongoDB is running.")
        print("   To start MongoDB on Windows, run: net start MongoDB")
        print("   To start MongoDB on macOS/Linux, run: sudo systemctl start mongod")
        return None, None
    except ServerSelectionTimeoutError:
        print("❌ MongoDB server selection timeout. Please check your connection.")
        return None, None
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return None, None

def create_collections(db):
    """Create necessary collections with proper indexes."""
    if db is None:
        return False
    
    try:
        print("\n📋 Creating collections and indexes...")
        
        # User Sessions Collection
        user_sessions = db.user_sessions
        user_sessions.create_index("user_id", unique=True)
        user_sessions.create_index("last_updated")
        user_sessions.create_index("session_id")
        print("✅ Created user_sessions collection with indexes")
        
        # User Chats Collection
        user_chats = db.user_chats
        user_chats.create_index([("user_id", 1), ("last_updated", -1)])
        user_chats.create_index("chat_session_id", unique=True)
        user_chats.create_index("user_session_id")
        print("✅ Created user_chats collection with indexes")
        
        # Product Catalog Collection (if needed)
        product_catalog = db.product_catalog
        product_catalog.create_index("product_id", unique=True)
        product_catalog.create_index("category")
        product_catalog.create_index("gender")
        product_catalog.create_index("price_pkr")
        print("✅ Created product_catalog collection with indexes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating collections: {e}")
        return False

def check_database_status(db):
    """Check the current status of the database."""
    if db is None:
        return
    
    try:
        print("\n📊 Database Status:")
        
        # Check collections
        collections = db.list_collection_names()
        print(f"   Collections found: {len(collections)}")
        for collection in collections:
            count = db[collection].count_documents({})
            print(f"   - {collection}: {count} documents")
        
        # Check user sessions
        user_sessions = db.user_sessions
        total_sessions = user_sessions.count_documents({})
        active_sessions = user_sessions.count_documents({
            "last_updated": {"$gte": datetime.datetime.now() - datetime.timedelta(hours=24)}
        })
        print(f"\n👥 User Sessions:")
        print(f"   Total sessions: {total_sessions}")
        print(f"   Active sessions (last 24h): {active_sessions}")
        
        # Check user chats
        user_chats = db.user_chats
        total_chats = user_chats.count_documents({})
        total_messages = sum([
            len(chat.get('messages', [])) for chat in user_chats.find({})
        ])
        print(f"\n💬 User Chats:")
        print(f"   Total chats: {total_chats}")
        print(f"   Total messages: {total_messages}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking database status: {e}")
        return False

def cleanup_old_data(db):
    """Clean up old sessions and chats."""
    if db is None:
        return False
    
    try:
        print("\n🧹 Cleaning up old data...")
        
        # Clean up old sessions (older than 7 days)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=7)
        old_sessions = db.user_sessions.count_documents({
            "last_updated": {"$lt": cutoff_date}
        })
        
        if old_sessions > 0:
            result = db.user_sessions.delete_many({
                "last_updated": {"$lt": cutoff_date}
            })
            print(f"   Deleted {result.deleted_count} old user sessions")
        else:
            print("   No old sessions to clean up")
        
        # Clean up old chats (older than 30 days)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
        old_chats = db.user_chats.count_documents({
            "last_updated": {"$lt": cutoff_date}
        })
        
        if old_chats > 0:
            result = db.user_chats.delete_many({
                "last_updated": {"$lt": cutoff_date}
            })
            print(f"   Deleted {result.deleted_count} old user chats")
        else:
            print("   No old chats to clean up")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning up old data: {e}")
        return False

def reset_database(db):
    """Reset the entire database (DANGEROUS - deletes all data)."""
    if db is None:
        return False
    
    try:
        print("\n🚨 RESET DATABASE")
        print("⚠️  WARNING: This will delete ALL data in the database!")
        
        confirm = input("Are you sure you want to reset the database? (yes/no): ").lower().strip()
        if confirm != 'yes':
            print("Database reset cancelled.")
            return False
        
        # Drop all collections
        collections = db.list_collection_names()
        for collection in collections:
            db[collection].drop()
            print(f"   Dropped collection: {collection}")
        
        # Recreate collections
        create_collections(db)
        
        print("✅ Database reset complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        return False

def export_database_stats(db, filename="database_stats.json"):
    """Export database statistics to a JSON file."""
    if db is None:
        return False
    
    try:
        print(f"\n📤 Exporting database stats to {filename}...")
        
        stats = {
            "export_date": datetime.datetime.now().isoformat(),
            "collections": {},
            "summary": {}
        }
        
        # Get collection stats
        collections = db.list_collection_names()
        for collection in collections:
            count = db[collection].count_documents({})
            stats["collections"][collection] = {
                "document_count": count,
                "indexes": list(db[collection].list_indexes())
            }
        
        # Get summary stats
        stats["summary"] = {
            "total_collections": len(collections),
            "total_documents": sum(stats["collections"][col]["document_count"] for col in stats["collections"]),
            "database_name": db.name
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"✅ Database stats exported to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting database stats: {e}")
        return False

def main():
    """Main function."""
    print("🔧 MongoDB Setup Script for Fashion Chatbot")
    print("=" * 50)
    
    # Connect to MongoDB
    db, client = connect_to_mongodb()
    if db is None:
        print("❌ Cannot proceed without database connection.")
        sys.exit(1)
    
    while True:
        print("\n📋 Available Operations:")
        print("1. Check database status")
        print("2. Create collections and indexes")
        print("3. Clean up old data")
        print("4. Export database stats")
        print("5. Reset database (DANGEROUS)")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            check_database_status(db)
        elif choice == '2':
            create_collections(db)
        elif choice == '3':
            cleanup_old_data(db)
        elif choice == '4':
            filename = input("Enter filename for export (default: database_stats.json): ").strip()
            if not filename:
                filename = "database_stats.json"
            export_database_stats(db, filename)
        elif choice == '5':
            reset_database(db)
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-6.")
    
    # Close connection
    if client:
        client.close()
        print("🔌 Database connection closed.")

if __name__ == "__main__":
    main()
