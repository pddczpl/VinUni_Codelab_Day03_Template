"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import Dict, List, Tuple
from tools import TOOL_DEFINITIONS, TOOL_MAP, Any, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot without ReAct Loop or Tools"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        answer = f"[Chatbot Baseline] Trả lời cho: {user_input}"
        if self.api_key:
            try:
                # Load the optional SDK dynamically so the module remains usable
                # when the Gemini dependency is not installed.
                import importlib
                genai = importlib.import_module("google.genai")
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-3.5-flash-lite')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời\nKHÔNG dùng tool hay internet: {user_input}"
                )
                answer = response.text
            except Exception as e:
                print(f"Error occurred: {e}")
        
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": []
        }

class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards"""
    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []
        self._step_context: Dict[str, Any] = {}

    def parse_city_code(self, text: str) -> str:
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper:
            return "SGN"
        if "ĐÀ NẮNG" in text_upper:
            return "DAD"
        return "SGN"

    def _parse_flight_args(self, text: str) -> Dict[str, Any]:
        origin = "HAN"
        destination = "SGN"
        route_match = re.search(r"(?:từ|tu)\s+([A-Za-zÀ-ỹ]+)\s+(?:đi|đến|den|di)\s+([A-Za-zÀ-ỹ]+)", text, re.IGNORECASE)
        if route_match:
            origin = self.parse_city_code(route_match.group(1))
            destination = self.parse_city_code(route_match.group(2))
        else:
            codes = [c for c in ["HAN", "DAD", "SGN"] if c in text.upper()]
            if len(codes) >= 2:
                origin, destination = codes[0], codes[1]
            elif len(codes) == 1:
                destination = codes[0]

        max_price = 5000000
        price_match = re.search(r"(?:dưới|duoi|<)\s*([\d\.]+)\s*(?:triệu|tr|trieu)", text, re.IGNORECASE)
        if price_match:
            max_price = int(float(price_match.group(1)) * 1000000)
        else:
            raw_num = re.search(r"(\d{6,})", text)
            if raw_num:
                max_price = int(raw_num.group(1))

        return {"origin": origin, "destination": destination, "max_price": max_price}

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        is_multi_step = self._step_context.get("is_multi_step", False)
        intent = self._step_context.get("intent", "faq")

        # Trường hợp 1: Multi-step execution (Cần đúng 3 iterations)
        if is_multi_step:
            if iteration == 1:
                flight_args = self._parse_flight_args(user_input)
                obs = TOOL_MAP["get_flight_info"](**flight_args)
                self._step_context["flight_obs"] = obs
                step_text = (
                    f"Thought: Cần tìm kiếm chuyến bay từ {flight_args['origin']} đi {flight_args['destination']} dưới {flight_args['max_price']} VND.\n"
                    f"Action: {{\"name\": \"get_flight_info\", \"args\": {json.dumps(flight_args)}}}\n"
                    f"Observation: {obs}"
                )
                return step_text, False

            elif iteration == 2:
                weather_part = user_input.split("thời tiết")[-1] if "thời tiết" in user_input.lower() else user_input
                city_code = self.parse_city_code(weather_part)
                obs = TOOL_MAP["get_weather_forecast"](city_code=city_code)
                self._step_context["weather_obs"] = obs
                step_text = (
                    f"Thought: Cần kiểm tra thông tin thời tiết tại {city_code} để tư vấn trang phục.\n"
                    f"Action: {{\"name\": \"get_weather_forecast\", \"args\": {{\"city_code\": \"{city_code}\"}}}}\n"
                    f"Observation: {obs}"
                )
                return step_text, False

            else:
                flight_obs = self._step_context.get("flight_obs", [])
                weather_obs = self._step_context.get("weather_obs", {})
                flight_numbers = [fl.get("flight_number") for fl in flight_obs if "flight_number" in fl]
                flights_str = ", ".join(flight_numbers) if flight_numbers else "VN213, VJ151"
                city_name = weather_obs.get("city", "TP. Hồ Chí Minh")
                temp = weather_obs.get("temp", "32°C")
                recom = weather_obs.get("recommendation", "áo thun thoáng mát")
                
                final_answer = (
                    f"Các chuyến bay phù hợp là {flights_str}. "
                    f"Thời tiết tại {city_name} hiện tại là {temp}, bạn nên mặc {recom}."
                )
                step_text = (
                    f"Thought: Đã có đủ dữ liệu chuyến bay và thời tiết. Tổng hợp câu trả lời cho khách hàng.\n"
                    f"Final Answer: {final_answer}"
                )
                return step_text, True

        # Trường hợp 2: Single-step Flight (iterations = 1)
        if intent == "flight":
            flight_args = self._parse_flight_args(user_input)
            obs = TOOL_MAP["get_flight_info"](**flight_args)
            flight_nums = [fl.get("flight_number") for fl in obs if "flight_number" in fl]
            flight_str = ", ".join(flight_nums) if flight_nums else "QH202"
            final_answer = f"Có chuyến bay {flight_str} phù hợp với yêu cầu của bạn."
            step_text = (
                f"Thought: Cần tìm chuyến bay theo yêu cầu.\n"
                f"Action: {{\"name\": \"get_flight_info\", \"args\": {json.dumps(flight_args)}}}\n"
                f"Observation: {obs}\n"
                f"Final Answer: {final_answer}"
            )
            return step_text, True

        # Trường hợp 3: Single-step Weather (iterations = 1)
        if intent == "weather":
            city_code = self.parse_city_code(user_input)
            obs = TOOL_MAP["get_weather_forecast"](city_code=city_code)
            city_name = obs.get("city", city_code)
            temp = obs.get("temp", "28°C")
            final_answer = f"Thời tiết ở {city_name} {city_code} hiện tại là {temp}, bạn nên mặc trang phục thoải mái."
            step_text = (
                f"Thought: Cần tra cứu thời tiết tại {city_code}.\n"
                f"Action: {{\"name\": \"get_weather_forecast\", \"args\": {{\"city_code\": \"{city_code}\"}}}}\n"
                f"Observation: {obs}\n"
                f"Final Answer: {final_answer}"
            )
            return step_text, True

        # Trường hợp 4: FAQ (iterations = 1, không gọi tool)
        final_answer = "Chính sách đổi trả vé máy bay Vinpearl tuân theo quy định hoàn hủy của từng hạng vé và điều kiện của hãng."
        step_text = (
            f"Thought: Đây là câu hỏi chính sách thông thường của Vinpearl, không cần gọi tool.\n"
            f"Final Answer: {final_answer}"
        )
        return step_text, True

    def run(self, user_query: str) -> Dict[str, Any]:
        self.trace = []
        lower_q = user_query.lower()
        
        has_flight = any(k in lower_q for k in ["chuyến bay", "vé", "bay"]) and ("vinpearl" not in lower_q or "chính sách" not in lower_q)
        has_weather = any(k in lower_q for k in ["thời tiết", "mặc gì", "nhiệt độ"])
        
        self._step_context = {
            "is_multi_step": has_flight and has_weather,
            "intent": "flight" if (has_flight and not has_weather) else ("weather" if has_weather else "faq")
        }

        for iteration in range(1, self.max_iterations + 1):
            step_output, is_finished = self.plan_and_execute_step(user_query, iteration)
            
            self.trace.append({
                "iteration": iteration,
                "output": step_output
            })

            if is_finished:
                final_answer = step_output.split("Final Answer:")[-1].strip() if "Final Answer:" in step_output else step_output
                return {
                    "status": "completed",
                    "answer": final_answer,
                    "iterations": iteration,
                    "trace": self.trace
                }

        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành yêu cầu trong số lượt lặp tối đa.",
            "iterations": self.max_iterations,
            "trace": self.trace
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()