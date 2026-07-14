import requests
import time
import json

# ================= 配置参数 =================
API_KEY = "REDACTED_BAIDU_API_KEY"  # TODO: 替换为您在千帆平台获取的真实API Key
VIDEO_URL = "https://link.jiyiho.cn/orfile/view.php/3222a4a890216588f9aa21acc036a474.mp4"  # TODO: 替换为支持直接下载的公网视频直链
# ============================================

# 1. 创建视频AI笔记任务
def create_ai_note_task(video_url):
    url = "https://qianfan.baidubce.com/v2/tools/ai_note/task_create"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "url": video_url
    }

    print("正在提交视频笔记任务...")
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("errno") == 0:
            task_id = result["data"]["task_id"]
            print(f"✅ 任务创建成功！Task ID: {task_id}")
            return task_id
        else:
            print(f"❌ 任务创建失败: {result.get('show_msg')}")
            return None
    else:
        print(f"❌ 请求异常，HTTP状态码: {response.status_code}")
        return None

# 2. 查询视频AI笔记任务结果
def query_ai_note_result(task_id):
    url = "https://qianfan.baidubce.com/v2/tools/ai_note/query"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    params = {
        "task_id": task_id
    }

    print(f"正在查询任务 {task_id} 的结果...")
    while True:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            result = response.json()
            errno = result.get("errno")
            
            if errno == 0:
                # 任务成功
                print("✅ 笔记生成成功！")
                return result["data"]
            elif errno == 10000:
                # 任务进行中，等待后重试
                print("⏳ 任务处理中，15秒后重试...")
                time.sleep(15)
            elif errno == 10001:
                # 任务失败
                print(f"❌ 任务处理失败: {result.get('show_msg')}")
                return None
            else:
                print(f"❌ 未知错误码: {errno}")
                return None
        else:
            print(f"❌ 查询请求异常，HTTP状态码: {response.status_code}")
            return None

# ================= 主程序入口 =================
if __name__ == "__main__":
    # 第一步：创建任务
    task_id = create_ai_note_task(VIDEO_URL)
    
    if task_id:
        # 第二步：轮询查询结果
        final_result = query_ai_note_result(task_id)
        
        if final_result:
            print("\n--- 最终笔记结果 ---")
            # 格式化输出结果，方便查看
            print(json.dumps(final_result, indent=2, ensure_ascii=False))
            
            # 提取并打印文稿笔记（tpl_no=1）和图文笔记（tpl_no=3），过滤掉大纲笔记
            note_list = final_result.get("list", [])
            for note in note_list:
                tpl_no = note.get("tpl_no")
                contents = note["detail"].get("contents", [])

                if tpl_no == "1":
                    print("\n--- 文稿笔记 ---")
                elif tpl_no == "3":
                    print("\n--- 图文笔记 ---")
                else:
                    # 跳过其他类型（如 tpl_no=2 大纲笔记）
                    continue

                for content in contents:
                    print(content)