"""
myuzePlay Search API using Pinecone
FREE Version - Using Sentence Transformers
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for GPT access

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "myuze-content"

# Initialize clients
print("🔧 Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("🤖 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ API Ready!\n")

# ========================================
# SEMANTIC SEARCH ENDPOINT
# ========================================

@app.route('/v1/search/semantic', methods=['POST'])
def semantic_search():
    """
    Semantic search endpoint
    
    Request body:
    {
        "query": "Hindi Bollywood video podcasts",
        "filters": {
            "ptype": ["Vodacast"],
            "country": "IN",
            "language": ["hi", "en"]
        },
        "top_k": 50,
        "score_threshold": 0.7
    }
    """
    try:
        data = request.json
        
        # Extract parameters
        query = data.get('query', '')
        filters = data.get('filters', {})
        top_k = data.get('top_k', 50)
        score_threshold = data.get('score_threshold', 0.7)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        print(f"🔍 Search query: {query}")
        
        # Create query embedding (FREE!)
        query_embedding = embedding_model.encode(query).tolist()
        
        # Build Pinecone filter
        pinecone_filter = build_filter(filters)
        
        # Search
        results = index.query(
            vector=query_embedding,
            filter=pinecone_filter,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            if match.score >= score_threshold:
                metadata = match.metadata
                
                # Parse comma-separated fields back to arrays
                item = {
                    "podcast_id": metadata.get('podcast_id'),
                    "score": float(match.score),
                    "title": metadata.get('title'),
                    "description": metadata.get('description'),
                    "ptype": metadata.get('ptype'),
                    "language": metadata.get('language', '').split(',') if metadata.get('language') else [],
                    "category": metadata.get('category'),
                    "category_levels": metadata.get('category_levels', '').split(',') if metadata.get('category_levels') else [],
                    "zoneid": metadata.get('zoneid', '').split(',') if metadata.get('zoneid') else [],
                    "is_billable": metadata.get('is_billable'),
                    "episode_count": metadata.get('episode_count'),
                    "ADDED_ON": metadata.get('ADDED_ON'),
                    "updated_at": metadata.get('updated_at')
                }
                
                formatted_results.append(item)
        
        print(f"✅ Found {len(formatted_results)} results")
        
        response = {
            "query": query,
            "filters_applied": filters,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========================================
# BUCKET GENERATION ENDPOINT
# ========================================

@app.route('/v1/buckets/generate', methods=['POST'])
def generate_buckets():
    """
    AI-powered bucket generation
    
    Request body:
    {
        "country": "IN",
        "num_buckets": 5,
        "bucket_size": 15
    }
    """
    try:
        data = request.json
        country = data.get('country', 'IN')
        num_buckets = data.get('num_buckets', 5)
        bucket_size = data.get('bucket_size', 15)
        
        print(f"🪣 Generating {num_buckets} buckets for {country}")
        
        # Predefined bucket queries by market
        bucket_templates = get_bucket_templates(country)
        
        buckets = []
        
        for template in bucket_templates[:num_buckets]:
            # Search for bucket contents (FREE embedding!)
            query_embedding = embedding_model.encode(template['query']).tolist()
            
            results = index.query(
                vector=query_embedding,
                filter=template.get('filter'),
                top_k=bucket_size,
                include_metadata=True
            )
            
            # Format bucket
            bucket = {
                "bucket_name": template['name'],
                "bucket_type": template['type'],
                "reasoning": template['reasoning'],
                "total_items": len(results.matches),
                "items": []
            }
            
            for match in results.matches:
                metadata = match.metadata
                bucket['items'].append({
                    "podcast_id": metadata.get('podcast_id'),
                    "title": metadata.get('title'),
                    "ptype": metadata.get('ptype'),
                    "language": metadata.get('language'),
                    "category": metadata.get('category'),
                    "score": float(match.score)
                })
            
            buckets.append(bucket)
            print(f"  ✅ {template['name']}: {len(results.matches)} items")
        
        return jsonify({
            "country": country,
            "buckets": buckets
        }), 200
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========================================
# HELPER FUNCTIONS
# ========================================

def build_filter(filters):
    """Build Pinecone filter from request filters"""
    pinecone_filter = {}
    
    if not filters:
        return None
    
    # Handle ptype filter
    if 'ptype' in filters:
        ptypes = filters['ptype']
        if isinstance(ptypes, list) and len(ptypes) == 1:
            pinecone_filter['ptype'] = {'$eq': ptypes[0]}
        elif isinstance(ptypes, list):
            pinecone_filter['ptype'] = {'$in': ptypes}
    
    # Handle country/zoneid filter
    if 'country' in filters:
        country = filters['country']
        # Need to search in comma-separated zoneid string
        pinecone_filter['$or'] = [
            {'zoneid': {'$eq': 'WorldWide'}},
            {'zoneid': {'$eq': country}}
        ]
    
    # Handle is_billable
    if 'monetization' in filters:
        monetization = filters['monetization']
        if isinstance(monetization, list):
            pinecone_filter['is_billable'] = {'$in': monetization}
    
    return pinecone_filter if pinecone_filter else None

def get_bucket_templates(country):
    """Get predefined bucket templates by country"""
    templates = {
        'IN': [
            {
                'name': 'Top Hindi Vodacasts',
                'type': 'Vodacast',
                'query': 'Hindi Bollywood entertainment celebrity video podcasts India',
                'filter': {'ptype': {'$eq': 'Vodacast'}},
                'reasoning': 'Hindi Vodacasts have highest engagement in India market'
            },
            {
                'name': 'Cricket Insider Talk',
                'type': 'Vodacast',
                'query': 'Cricket IPL T20 sports analysis commentary India',
                'filter': {'ptype': {'$eq': 'Vodacast'}},
                'reasoning': 'Cricket is the most popular sport in India'
            },
            {
                'name': 'Startup Stories India',
                'type': 'Show',
                'query': 'Business entrepreneur startup founder success stories India',
                'filter': {'ptype': {'$eq': 'Show'}},
                'reasoning': 'Strong startup ecosystem interest'
            },
            {
                'name': 'New Hindi Podcasts',
                'type': 'Podcast',
                'query': 'Hindi entertainment news talk podcasts recently added',
                'filter': {'ptype': {'$eq': 'Podcast'}},
                'reasoning': 'Recency-driven discovery for Hindi listeners'
            },
            {
                'name': 'Mythology Audiobooks India',
                'type': 'Book',
                'query': 'Indian mythology Ramayana Mahabharata Hindu epics audiobooks',
                'filter': {'ptype': {'$eq': 'Book'}},
                'reasoning': 'Cultural storytelling interest'
            },
            {
                'name': 'Comedy & Standup Shows',
                'type': 'Vodacast',
                'query': 'Hindi comedy standup funny entertainment India',
                'filter': {'ptype': {'$eq': 'Vodacast'}},
                'reasoning': 'Growing comedy scene in India'
            },
            {
                'name': 'Technology & Gadgets',
                'type': 'Show',
                'query': 'Technology mobile phones gadgets tech reviews India',
                'filter': {'ptype': {'$eq': 'Show'}},
                'reasoning': 'High tech adoption market'
            }
        ],
        'PK': [
            {
                'name': 'Top Urdu Vodacasts',
                'type': 'Vodacast',
                'query': 'Urdu entertainment talk video podcasts Pakistan',
                'filter': {'ptype': {'$eq': 'Vodacast'}},
                'reasoning': 'Urdu is primary language in Pakistan'
            },
            {
                'name': 'Cricket Talk Pakistan',
                'type': 'Podcast',
                'query': 'Cricket PSL Pakistan sports commentary Urdu',
                'filter': {'ptype': {'$eq': 'Podcast'}},
                'reasoning': 'Cricket dominates sports interest'
            },
            {
                'name': 'Islamic Content',
                'type': 'Podcast',
                'query': 'Islamic Quran Hadith religious spiritual Pakistan Urdu',
                'filter': {'ptype': {'$eq': 'Podcast'}},
                'reasoning': 'Strong religious content interest'
            }
        ],
        'US': [
            {
                'name': 'Top English Podcasts',
                'type': 'Podcast',
                'query': 'English talk entertainment news podcasts USA',
                'filter': {'ptype': {'$eq': 'Podcast'}},
                'reasoning': 'English is primary language'
            }
        ]
    }
    
    return templates.get(country, templates['IN'])

# ========================================
# HEALTH CHECK
# ========================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        stats = index.describe_index_stats()
        return jsonify({
            "status": "healthy",
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "embedding_model": "all-MiniLM-L6-v2",
            "cost": "FREE"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# ========================================
# TEST ENDPOINT
# ========================================

@app.route('/test', methods=['GET'])
def test():
    """Quick test endpoint"""
    try:
        # Test search
        test_query = "Hindi Bollywood entertainment"
        query_embedding = embedding_model.encode(test_query).tolist()
        
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
        return jsonify({
            "test_query": test_query,
            "results_found": len(results.matches),
            "sample_results": [
                {
                    "title": m.metadata.get('title'),
                    "score": float(m.score)
                } for m in results.matches
            ]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================================
# RUN SERVER
# ========================================

if __name__ == '__main__':
    print("="*70)
    print("🚀 myuzePlay Search API (FREE Version)")
    print("="*70)
    print()
    print("📍 Endpoints:")
    print("   POST /v1/search/semantic - Semantic search")
    print("   POST /v1/buckets/generate - Generate buckets")
    print("   GET  /health - Health check")
    print("   GET  /test - Quick test")
    print()
    print("💰 Cost: $0 (Completely FREE!)")
    print()
    
    # For development
    app.run(host='0.0.0.0', port=5000, debug=True)
    
    # For production, use gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app