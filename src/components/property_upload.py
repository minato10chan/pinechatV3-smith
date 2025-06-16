import streamlit as st
from src.services.pinecone_service import PineconeService
import pandas as pd
import json
import traceback
from datetime import datetime
import tiktoken

# 都道府県と市区町村のデータ
PREFECTURES = [
    "埼玉県", "千葉県", "東京都", "神奈川県"
]

# 主要な市区町村のデータ（例として東京都の区を記載）
CITIES = {
    "埼玉県": [
        "川越市", "さいたま市"
    ],
    # 他の都道府県の市区町村も同様に追加可能
}

def split_property_data(property_data: dict, max_tokens: int = 3000) -> list:
    """物件データを複数のチャンクに分割する"""
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    
    # 基本情報（常に含める）
    base_info = {
        "property_name": property_data["property_name"],
        "property_type": property_data["property_type"],
        "prefecture": property_data["prefecture"],
        "city": property_data["city"],
        "detailed_address": property_data["detailed_address"],
        "latitude": property_data.get("latitude", "0.0"),
        "longitude": property_data.get("longitude", "0.0")
    }
    
    # 詳細情報を分割
    details = property_data.get("property_details", "")
    if not details:
        return [{"text": json.dumps(base_info, ensure_ascii=False), "metadata": base_info}]
    
    # 詳細情報を段落で分割
    paragraphs = [p.strip() for p in details.split('\n') if p.strip()]
    
    # 段落を意味のある単位でグループ化
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        # 段落のトークン数を計算
        paragraph_tokens = len(encoding.encode(paragraph))
        
        # 現在のチャンクに追加した場合の長さを計算
        if current_chunk:
            test_text = "\n".join(current_chunk + [paragraph])
        else:
            test_text = paragraph
        
        test_tokens = len(encoding.encode(test_text))
        
        # チャンクの長さが制限を超える場合、新しいチャンクを開始
        if test_tokens > max_tokens:
            if current_chunk:
                # 現在のチャンクを保存
                chunk_info = base_info.copy()
                chunk_info["property_details"] = "\n".join(current_chunk)
                chunk_info["chunk_number"] = len(chunks) + 1
                
                # チャンクのトークン数を確認
                chunk_text = json.dumps(chunk_info, ensure_ascii=False)
                chunk_tokens = len(encoding.encode(chunk_text))
                
                if chunk_tokens > max_tokens:
                    # チャンクが大きすぎる場合は、さらに小さく分割
                    sub_chunks = []
                    current_sub_chunk = []
                    current_sub_length = 0
                    
                    for p in current_chunk:
                        p_tokens = len(encoding.encode(p))
                        if current_sub_length + p_tokens > max_tokens // 2:
                            if current_sub_chunk:
                                sub_chunks.append(current_sub_chunk)
                            current_sub_chunk = [p]
                            current_sub_length = p_tokens
                        else:
                            current_sub_chunk.append(p)
                            current_sub_length += p_tokens
                    
                    if current_sub_chunk:
                        sub_chunks.append(current_sub_chunk)
                    
                    # サブチャンクごとにチャンクを作成
                    for j, sub_group in enumerate(sub_chunks):
                        sub_chunk_info = base_info.copy()
                        sub_chunk_info["property_details"] = "\n".join(sub_group)
                        sub_chunk_info["chunk_number"] = len(chunks) + 1
                        sub_chunk_info["sub_chunk_number"] = j + 1
                        sub_chunk_info["total_sub_chunks"] = len(sub_chunks)
                        
                        chunk = {
                            "text": json.dumps(sub_chunk_info, ensure_ascii=False),
                            "metadata": sub_chunk_info
                        }
                        chunks.append(chunk)
                else:
                    chunk = {
                        "text": chunk_text,
                        "metadata": chunk_info
                    }
                    chunks.append(chunk)
            
            # 新しいチャンクを開始
            current_chunk = [paragraph]
            current_length = paragraph_tokens
        else:
            current_chunk.append(paragraph)
            current_length = test_tokens
    
    # 最後のチャンクを処理
    if current_chunk:
        chunk_info = base_info.copy()
        chunk_info["property_details"] = "\n".join(current_chunk)
        chunk_info["chunk_number"] = len(chunks) + 1
        
        # チャンクのトークン数を確認
        chunk_text = json.dumps(chunk_info, ensure_ascii=False)
        chunk_tokens = len(encoding.encode(chunk_text))
        
        if chunk_tokens > max_tokens:
            # チャンクが大きすぎる場合は、さらに小さく分割
            sub_chunks = []
            current_sub_chunk = []
            current_sub_length = 0
            
            for p in current_chunk:
                p_tokens = len(encoding.encode(p))
                if current_sub_length + p_tokens > max_tokens // 2:
                    if current_sub_chunk:
                        sub_chunks.append(current_sub_chunk)
                    current_sub_chunk = [p]
                    current_sub_length = p_tokens
                else:
                    current_sub_chunk.append(p)
                    current_sub_length += p_tokens
            
            if current_sub_chunk:
                sub_chunks.append(current_sub_chunk)
            
            # サブチャンクごとにチャンクを作成
            for j, sub_group in enumerate(sub_chunks):
                sub_chunk_info = base_info.copy()
                sub_chunk_info["property_details"] = "\n".join(sub_group)
                sub_chunk_info["chunk_number"] = len(chunks) + 1
                sub_chunk_info["sub_chunk_number"] = j + 1
                sub_chunk_info["total_sub_chunks"] = len(sub_chunks)
                
                chunk = {
                    "text": json.dumps(sub_chunk_info, ensure_ascii=False),
                    "metadata": sub_chunk_info
                }
                chunks.append(chunk)
        else:
            chunk = {
                "text": chunk_text,
                "metadata": chunk_info
            }
            chunks.append(chunk)
    
    # 総チャンク数を更新
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    return chunks

def render_property_upload(pinecone_service: PineconeService):
    """物件情報のアップロードUIを表示"""
    st.title("🏠 物件情報のアップロード")
    
    with st.form("property_upload_form"):
        st.markdown("### 物件情報の入力")
        
        # 物件名
        property_name = st.text_input("物件名", help="物件の名称を入力してください")
        
        # 物件種別
        property_type = st.selectbox(
            "物件種別",
            ["一戸建て", "土地", "マンション"],
            help="物件の種別を選択してください"
        )
        
        # 都道府県と市区町村の選択
        col1, col2 = st.columns(2)
        with col1:
            prefecture = st.selectbox(
                "都道府県",
                PREFECTURES,
                help="物件の所在地の都道府県を選択してください"
            )
        
        with col2:
            # 選択された都道府県に基づいて市区町村のリストを表示
            cities = CITIES.get(prefecture, [])
            city = st.selectbox(
                "市区町村",
                cities,
                help="物件の所在地の市区町村を選択してください"
            )
        
        # 詳細住所
        detailed_address = st.text_input("詳細住所", help="物件の詳細な住所を入力してください")
        
        # 物件の詳細情報
        property_details = st.text_area(
            "物件の詳細情報",
            help="物件の詳細な情報を入力してください（長い文章は自動的に分割されます）"
        )
        
        # 緯度・経度
        col3, col4 = st.columns(2)
        with col3:
            latitude = st.text_input(
                "緯度",
                value="0.0",
                help="物件の緯度を入力してください"
            )
        with col4:
            longitude = st.text_input(
                "経度",
                value="0.0",
                help="物件の経度を入力してください"
            )
        
        # アップロードボタン
        submit_button = st.form_submit_button("アップロード")
        
        if submit_button:
            try:
                # 必須項目のチェック
                if not all([property_name, property_type, prefecture, city]):
                    st.error("❌ 必須項目（物件名、物件種別、都道府県、市区町村）を入力してください")
                    return
                
                # 物件情報の構造化
                property_data = {
                    "property_name": property_name,
                    "property_type": property_type,
                    "prefecture": prefecture,
                    "city": city,
                    "detailed_address": detailed_address,
                    "property_details": property_details,
                    "latitude": latitude,
                    "longitude": longitude
                }
                
                # 物件データをチャンクに分割
                chunks = split_property_data(property_data)
                
                # チャンクごとにIDを付与
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                for i, chunk in enumerate(chunks):
                    chunk["id"] = f"property_{timestamp}_{i}"
                
                # Pineconeへのアップロード
                pinecone_service.upload_chunks(chunks, namespace="property")
                
                st.success(f"✅ 物件情報を{len(chunks)}件のチャンクに分割してアップロードしました")
                
            except Exception as e:
                st.error(f"❌ アップロードに失敗しました: {str(e)}")
                st.error(f"🔍 エラーの詳細: {type(e).__name__}")
                st.error(f"📜 スタックトレース:\n{traceback.format_exc()}") 