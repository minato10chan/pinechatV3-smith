from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os
import tiktoken
from openai import OpenAI
from ..config.settings import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    OPENAI_API_KEY,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_RESPONSE_TEMPLATE
)

class LangChainService:
    def __init__(self, callback_manager=None):
        """LangChainサービスの初期化"""
        # OpenAIクライアントの初期化
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # チャットモデルの初期化
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
             # model_name="gpt-4o-mini",
            model_name="gpt-3.5-turbo",
            temperature=0.85,
            callback_manager=callback_manager
        )
        
        # 埋め込みモデルの初期化
        self.embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model="text-embedding-ada-002"
        )
        
        # トークンカウンターの初期化
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # PineconeのAPIキーを環境変数に設定
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        
        # Pineconeベクトルストアの初期化
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embeddings
        )
        
        # チャット履歴の初期化
        self.message_history = ChatMessageHistory()
        
        # デフォルトのプロンプトテンプレート
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.response_template = DEFAULT_RESPONSE_TEMPLATE

    def check_api_usage(self):
        """OpenAI APIの使用状況を確認"""
        try:
            # 使用状況の取得
            # usage = self.openai_client.usage.retrieve()
            
            # 使用状況の表示
            print("\n=== OpenAI API Usage ===")
            # print(f"Total Tokens: {usage.total_tokens}")
            # print(f"Total Cost: ${usage.total_cost:.4f}")
            # print(f"Usage Period: {usage.period}")
            
            # クォータ情報の取得
            # quota = self.openai_client.quota.retrieve()
            print("\n=== OpenAI API Quota ===")
            # print(f"Total Quota: ${quota.total_quota:.2f}")
            # print(f"Used Quota: ${quota.used_quota:.2f}")
            # print(f"Remaining Quota: ${quota.remaining_quota:.2f}")
            # print(f"Quota Period: {quota.period}")
            
            # 警告メッセージ
            # if quota.remaining_quota < 1.0:
            #     print("\n⚠️ Warning: Remaining quota is less than $1.0")
            # if quota.remaining_quota < 0.1:
            #     print("🚨 Critical: Remaining quota is less than $0.1")
                
        except Exception as e:
            error_message = str(e)
            print(f"\n❌ Error checking API usage: {error_message}")
            
            if "insufficient_quota" in error_message:
                print("\n🚨 Critical: API quota has been exceeded!")
                print("Please check your OpenAI API key and billing settings.")
                print("You can check your usage and quota at: https://platform.openai.com/account/usage")
            elif "object has no attribute" in error_message:
                print("\n⚠️ Warning: Unable to check API usage. This might be due to API changes or permissions.")
                print("Please check your OpenAI API key and ensure it has the necessary permissions.")
            else:
                print("\n⚠️ Warning: Unable to check API usage. Please verify your API key and permissions.")

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        return len(self.encoding.encode(text))

    def get_relevant_context(self, query: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, List[Dict[str, Any]]]:
        """クエリに関連する文脈を取得"""
        try:
            # クエリのトークン数をカウント
            query_tokens = self.count_tokens(query)
            print(f"クエリのトークン数: {query_tokens}")
            
            # クエリのベクトル化
            query_vector = self.embeddings.embed_query(query)
            
            # より多くの結果を取得して、後でフィルタリング
            docs = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)
            
            # メタデータも検索対象に含める
            for doc in docs:
                # メタデータの各フィールドを検索対象に追加
                metadata_text = []
                for key, value in doc[0].metadata.items():
                    if isinstance(value, str):
                        # メタデータの値をテキストに追加
                        metadata_text.append(f"{key}: {value}")
                
                # メタデータをテキストの前に追加
                if metadata_text:
                    doc[0].page_content = "\n".join(metadata_text) + "\n\n" + doc[0].page_content
            
            # スコアでフィルタリング
            filtered_docs = []
            for doc in docs:
                if doc[1] >= SIMILARITY_THRESHOLD:
                    filtered_docs.append(doc)
                    if len(filtered_docs) >= top_k:
                        break
            
            # フィルタリング後の結果が0件の場合は、スコアに関係なく上位K件を使用
            if not filtered_docs and docs:
                filtered_docs = docs[:top_k]
            
            context_text = "\n".join([doc[0].page_content for doc in filtered_docs])
            
            # コンテキストのトークン数をカウント
            context_tokens = self.count_tokens(context_text)
            print(f"コンテキストのトークン数: {context_tokens}")
            
            search_details = []
            for doc in filtered_docs:
                detail = {
                    "スコア": round(doc[1], 4),  # 類似度スコアを小数点4桁まで表示
                    "テキスト": doc[0].page_content[:100] + "...",  # テキストの一部を表示
                    "メタデータ": doc[0].metadata,  # メタデータを追加
                    "類似度判断": {
                        "閾値": SIMILARITY_THRESHOLD,
                        "閾値以上": doc[1] >= SIMILARITY_THRESHOLD,
                        "スコア詳細": f"スコア {round(doc[1], 4)} は閾値 {SIMILARITY_THRESHOLD} に対して {'以上' if doc[1] >= SIMILARITY_THRESHOLD else '未満'}",
                        "理解過程": {
                            "クエリ": query,
                            "テキスト": doc[0].page_content,
                            "類似度計算": {
                                "スコア": round(doc[1], 4)
                            }
                        }
                    }
                }
                search_details.append(detail)
            
            print(f"検索クエリ: {query}")  # デバッグ用
            print(f"検索結果数: {len(filtered_docs)}")  # デバッグ用
            for detail in search_details:
                print(f"スコア: {detail['スコア']}, テキスト: {detail['テキスト']}")  # デバッグ用
            
            return context_text, search_details
            
        except Exception as e:
            error_message = str(e)
            if "insufficient_quota" in error_message:
                print("\n🚨 Critical: API quota has been exceeded!")
                print("Please check your OpenAI API key and billing settings.")
                print("You can check your usage and quota at: https://platform.openai.com/account/usage")
                # 空のコンテキストとエラー詳細を返す
                return "", [{
                    "エラー": True,
                    "エラーメッセージ": "API quota has been exceeded",
                    "エラータイプ": "API Quota Error",
                    "推奨アクション": "Please update your API key in Streamlit Cloud settings"
                }]
            else:
                print(f"\n❌ Error in get_relevant_context: {error_message}")
                return "", [{
                    "エラー": True,
                    "エラーメッセージ": error_message,
                    "エラータイプ": "Unknown Error"
                }]

    def get_response(self, query: str, system_prompt: str = None, response_template: str = None, property_info: str = None, chat_history: list = None) -> Tuple[str, Dict[str, Any]]:
        """クエリに対する応答を生成"""
        try:
            # プロンプトの設定
            system_prompt = system_prompt or self.system_prompt
            response_template = response_template or self.response_template
            
            # メッセージリストの作成
            messages = [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", "参照文脈:\n{context}")
            ]
            
            # 物件情報がある場合は追加
            if property_info:
                messages.append(("system", "物件情報:\n{property_info}"))
            
            # ユーザー入力の追加
            messages.append(("human", "{input}"))
            
            # プロンプトテンプレートの設定
            prompt = ChatPromptTemplate.from_messages(messages)
            
            # チェーンの初期化
            chain = prompt | self.llm
            
            # 関連する文脈を取得
            context, search_details = self.get_relevant_context(query)
            
            # チャット履歴を設定
            if chat_history:
                self.message_history.messages = []
                for role, content in chat_history:
                    if role == "human":
                        self.message_history.add_user_message(content)
                    elif role == "ai":
                        self.message_history.add_ai_message(content)
            
            # 会話履歴を最適化
            self.optimize_chat_history()
            
            # プロンプトのトークン数をカウント
            prompt_tokens = self.count_tokens(system_prompt)
            print(f"システムプロンプトのトークン数: {prompt_tokens}")
            
            # チャット履歴のトークン数をカウント
            history_tokens = sum(self.count_tokens(msg.content) for msg in self.message_history.messages)
            print(f"チャット履歴のトークン数: {history_tokens}")
            
            # 応答を生成
            response = chain.invoke({
                "chat_history": self.message_history.messages,
                "context": context,
                "property_info": property_info or "物件情報はありません。",
                "input": query
            })
            
            # 応答のトークン数をカウント
            response_tokens = self.count_tokens(response.content)
            print(f"応答のトークン数: {response_tokens}")
            
            # メッセージを履歴に追加
            self.message_history.add_user_message(query)
            self.message_history.add_ai_message(response.content)
            
            # 詳細情報の作成
            details = {
                "モデル": "gpt-3.5-turbo",
                "会話履歴": "有効",
                "トークン数": {
                    "クエリ": self.count_tokens(query),
                    "システムプロンプト": prompt_tokens,
                    "チャット履歴": history_tokens,
                    "コンテキスト": self.count_tokens(context),
                    "応答": response_tokens,
                    "合計": prompt_tokens + history_tokens + self.count_tokens(context) + response_tokens
                },
                "文脈検索": {
                    "検索結果数": len(search_details),
                    "マッチしたチャンク": search_details
                },
                "プロンプト": {
                    "システムプロンプト": system_prompt,
                    "応答テンプレート": response_template
                },
                "物件情報": property_info or "物件情報はありません。",
                "会話履歴数": len(chat_history) if chat_history else 0
            }
            
            return response.content, details
            
        except Exception as e:
            error_message = str(e)
            if "insufficient_quota" in error_message:
                error_response = "申し訳ありません。APIの利用制限に達しました。\n\n" + \
                               "以下の手順で対応をお願いします：\n" + \
                               "1. OpenAIのアカウント設定を確認してください\n" + \
                               "2. 新しいAPIキーを取得してください\n" + \
                               "3. Streamlit Cloudの設定で新しいAPIキーを更新してください\n\n" + \
                               "詳細はこちらで確認できます：\n" + \
                               "https://platform.openai.com/account/usage"
            else:
                error_response = f"エラーが発生しました：{error_message}"
            
            error_details = {
                "エラー": True,
                "エラーメッセージ": error_message,
                "エラータイプ": "API Quota Error" if "insufficient_quota" in error_message else "Unknown Error"
            }
            
            return error_response, error_details

    def optimize_chat_history(self, max_tokens: int = 12000) -> None:
        """会話履歴を最適化し、重要なメッセージのみを保持"""
        if not self.message_history.messages:
            return

        # 現在のトークン数を計算
        current_tokens = sum(self.count_tokens(msg.content) for msg in self.message_history.messages)
        
        # トークン数が制限を超えていない場合は何もしない
        if current_tokens <= max_tokens:
            return

        # メッセージを重要度で分類
        important_messages = []
        other_messages = []
        
        for msg in self.message_history.messages:
            # システムメッセージは常に保持
            if isinstance(msg, SystemMessage):
                important_messages.append(msg)
                continue
                
            # 最後のNメッセージは保持（Nは設定可能）
            if len(important_messages) < 4:  # 最後の4メッセージを保持
                important_messages.append(msg)
                continue
                
            # その他のメッセージは一時的に保存
            other_messages.append(msg)

        # 重要メッセージのトークン数を計算
        important_tokens = sum(self.count_tokens(msg.content) for msg in important_messages)
        
        # 残りのトークン数
        remaining_tokens = max_tokens - important_tokens
        
        # 残りのトークン数に基づいて、他のメッセージを追加
        for msg in reversed(other_messages):
            msg_tokens = self.count_tokens(msg.content)
            if msg_tokens <= remaining_tokens:
                important_messages.insert(0, msg)  # 先頭に追加
                remaining_tokens -= msg_tokens
            else:
                break

        # 最適化されたメッセージで履歴を更新
        self.message_history.messages = important_messages

    def clear_memory(self):
        """会話メモリをクリア"""
        self.message_history.clear() 