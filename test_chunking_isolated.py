#!/usr/bin/env python3
"""
Completely isolated test for chunking logic without any Streamlit dependencies
"""
import json
import sys
import tiktoken

def split_property_data_isolated(property_data: dict, max_tokens: int = 8000) -> list:
    """Isolated version of property data splitting function"""
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    
    base_info = {
        "property_name": property_data["property_name"],
        "property_type": property_data["property_type"],
        "prefecture": property_data["prefecture"],
        "city": property_data["city"],
        "detailed_address": property_data["detailed_address"],
        "latitude": property_data.get("latitude", "0.0"),
        "longitude": property_data.get("longitude", "0.0")
    }
    
    details = property_data.get("property_details", "")
    if not details:
        return [{"text": json.dumps(base_info, ensure_ascii=False), "metadata": base_info}]
    
    paragraphs = [p.strip() for p in details.split('\n') if p.strip()]
    
    split_paragraphs = []
    for paragraph in paragraphs:
        paragraph_tokens = len(encoding.encode(paragraph))
        
        if paragraph_tokens <= max_tokens:
            split_paragraphs.append(paragraph)
        else:
            sentences = [s.strip() for s in paragraph.replace('。', '。\n').split('\n') if s.strip()]
            current_sentence_group = []
            current_group_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = len(encoding.encode(sentence))
                
                if current_group_tokens + sentence_tokens > max_tokens:
                    if current_sentence_group:
                        split_paragraphs.append(''.join(current_sentence_group))
                    current_sentence_group = [sentence]
                    current_group_tokens = sentence_tokens
                else:
                    current_sentence_group.append(sentence)
                    current_group_tokens += sentence_tokens
            
            if current_sentence_group:
                split_paragraphs.append(''.join(current_sentence_group))
    
    chunks = []
    current_chunk = []
    
    for paragraph in split_paragraphs:
        if current_chunk:
            test_text = "\n".join(current_chunk + [paragraph])
        else:
            test_text = paragraph
        
        test_tokens = len(encoding.encode(test_text))
        
        if test_tokens > max_tokens:
            if current_chunk:
                chunk_info = base_info.copy()
                chunk_info["property_details"] = "\n".join(current_chunk)
                chunk_info["chunk_number"] = len(chunks) + 1
                
                chunk_text = json.dumps(chunk_info, ensure_ascii=False)
                chunk = {
                    "text": chunk_text,
                    "metadata": chunk_info
                }
                chunks.append(chunk)
            
            current_chunk = [paragraph]
        else:
            current_chunk.append(paragraph)
    
    if current_chunk:
        chunk_info = base_info.copy()
        chunk_info["property_details"] = "\n".join(current_chunk)
        chunk_info["chunk_number"] = len(chunks) + 1
        
        chunk_text = json.dumps(chunk_info, ensure_ascii=False)
        chunk = {
            "text": chunk_text,
            "metadata": chunk_info
        }
        chunks.append(chunk)
    
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)
    
    return chunks

def test_isolated_chunking():
    """Test chunking functionality in complete isolation"""
    print("=== Isolated Property Text Chunking Test ===")
    
    try:
        long_description = """
この物件は東京都心部に位置する高級マンションです。最寄り駅から徒歩3分という抜群の立地条件を誇り、都心へのアクセスが非常に便利です。建物は地上20階建ての鉄筋コンクリート造で、2022年に竣工した新築物件です。

外観は洗練されたモダンデザインを採用し、エントランスには24時間有人管理のコンシェルジュサービスが配置されています。セキュリティ面では、オートロック、防犯カメラ、宅配ボックスなど最新の設備が完備されています。

各住戸は南向きの角部屋を中心とした設計で、大きな窓から豊富な自然光が差し込みます。室内は高級感のある内装仕上げで、システムキッチン、浴室乾燥機、床暖房、エアコンなどの設備が標準装備されています。

共用施設として、屋上庭園、フィットネスジム、ゲストルーム、キッズルーム、パーティールームなどが用意されており、住民の快適な生活をサポートします。また、敷地内には来客用駐車場も完備されています。

周辺環境は非常に充実しており、徒歩圏内にスーパーマーケット、コンビニエンスストア、銀行、郵便局、病院、薬局などの生活に必要な施設が揃っています。また、近隣には有名な公園があり、緑豊かな環境でリラックスできます。

教育環境も優れており、評判の良い小学校、中学校が学区内にあります。また、有名私立学校へのアクセスも良好で、子育て世代にとって理想的な環境です。保育園や幼稚園も複数あり、待機児童の心配も少ない地域です。

交通アクセスは複数路線が利用可能で、主要ターミナル駅まで乗り換えなしでアクセスできます。朝の通勤ラッシュ時でも比較的混雑が少なく、快適な通勤が可能です。また、羽田空港や成田空港へのアクセスも良好で、出張や旅行の際にも便利です。

商業施設も充実しており、大型ショッピングモール、デパート、専門店街などが近隣にあります。レストラン、カフェ、居酒屋なども多数あり、外食やエンターテイメントにも困りません。

将来性についても、この地域は再開発計画が進行中で、さらなる発展が期待されています。新しい商業施設や公共施設の建設も予定されており、資産価値の向上も見込まれます。

管理体制は信頼できる大手管理会社が担当し、建物の維持管理、清掃、設備点検などが適切に行われています。管理費や修繕積立金も適正な水準に設定されており、長期的な資産価値の維持が期待できます。

この物件の最大の魅力は、都心部でありながら静かで落ち着いた住環境を提供していることです。近隣には緑豊かな公園や散歩道があり、都市生活の中でも自然を感じることができます。また、建物の設計にも環境への配慮が見られ、省エネルギー設備や太陽光発電システムなどが導入されています。

住戸の間取りは多様で、単身者向けの1Kから家族向けの3LDKまで幅広く用意されています。各住戸には最新の設備が完備されており、快適な生活を送ることができます。特に、キッチンは高級ブランドの設備を使用し、料理好きの方にも満足していただけるでしょう。

バスルームも広々とした設計で、一日の疲れを癒すことができます。浴室乾燥機や追い焚き機能なども標準装備されており、快適なバスタイムを楽しむことができます。

収納スペースも豊富に用意されており、季節の衣類や日用品などをすっきりと整理することができます。ウォークインクローゼットや床下収納など、様々なタイプの収納が配置されています。

共用部分の管理も行き届いており、エントランスホールは常に清潔に保たれています。宅配ボックスも大型のものが設置されており、不在時でも安心して荷物を受け取ることができます。

駐車場は機械式と平面式の両方が用意されており、車をお持ちの方にも便利です。また、自転車置き場やバイク置き場も完備されており、様々な交通手段に対応しています。

ペット飼育についても相談可能で、愛犬や愛猫と一緒に暮らすことができます。近隣にはペット病院やペットショップもあり、ペットとの生活をサポートする環境が整っています。

インターネット環境も充実しており、光ファイバーが各住戸まで引き込まれています。在宅ワークやオンライン学習にも対応できる高速インターネット環境が整備されています。

防災面でも安心で、建物は最新の耐震基準に適合しており、非常用発電機や防災備蓄倉庫なども完備されています。定期的な防災訓練も実施されており、住民の安全意識も高く保たれています。
""" * 20  # Make it 20x longer to test chunking thoroughly
        
        test_property = {
            'property_name': 'テスト統合物件（完全分離版）',
            'property_type': 'マンション',
            'prefecture': '東京都',
            'city': '渋谷区',
            'detailed_address': '渋谷1-1-1',
            'property_details': long_description,
            'latitude': '35.6580',
            'longitude': '139.7016'
        }
        
        print(f"Property details length: {len(long_description):,} characters")
        
        for token_limit in [2000, 8000]:
            print(f"\n--- Testing with {token_limit} token limit ---")
            chunks = split_property_data_isolated(test_property, max_tokens=token_limit)
            print(f"✅ Created {len(chunks)} chunks with {token_limit} token limit")
            
            all_valid = True
            total_size = 0
            max_chunk_size = 0
            
            for i, chunk in enumerate(chunks):
                chunk_json = chunk["text"]
                chunk_bytes = chunk_json.encode('utf-8')
                chunk_size = len(chunk_bytes)
                total_size += chunk_size
                max_chunk_size = max(max_chunk_size, chunk_size)
                
                print(f"  Chunk {i+1}: {chunk_size:,} bytes")
                
                if chunk_size > 40 * 1024:
                    print(f"  ❌ Chunk {i+1} exceeds 40KB limit!")
                    all_valid = False
                else:
                    print(f"  ✅ Chunk {i+1} is within 40KB limit")
                
                try:
                    metadata = chunk["metadata"]
                    assert "property_name" in metadata
                    assert "chunk_number" in metadata
                    assert "total_chunks" in metadata
                    print(f"  ✅ Chunk {i+1} has valid metadata structure")
                except Exception as e:
                    print(f"  ❌ Chunk {i+1} has invalid metadata: {e}")
                    all_valid = False
            
            print(f"Total size of all chunks: {total_size:,} bytes")
            print(f"Largest chunk size: {max_chunk_size:,} bytes")
            print(f"40KB limit: {40 * 1024:,} bytes")
            
            if all_valid:
                print(f"🎉 All chunks are valid with {token_limit} token limit!")
            else:
                print(f"❌ Some chunks are invalid with {token_limit} token limit!")
        
        return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_isolated_chunking()
    sys.exit(0 if success else 1)
