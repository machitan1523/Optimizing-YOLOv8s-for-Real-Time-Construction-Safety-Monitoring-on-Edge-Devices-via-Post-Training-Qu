import os
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np


def get_real_power():
    try:
        
        result = subprocess.run(['vcgencmd', 'pmic_read_adc'], stdout=subprocess.PIPE, text=True)
        output = result.stdout
       
        currents = {}
        volts = {}
       
        
        for line in output.split('\n'):
            line = line.strip()
            if not line: continue
           
            parts = line.split('=')
            if len(parts) < 2: continue
           
            name_part = parts[0].split()[0]
            value_part = parts[1].replace('V', '').replace('A', '')
           
            try:
                val = float(value_part)
                if name_part.endswith('_A'):
                    currents[name_part[:-2]] = val
                elif name_part.endswith('_V'):
                    volts[name_part[:-2]] = val
            except: continue

        
        total_internal_power = 0.0
        for name, current in currents.items():
            if name in volts:
                total_internal_power += (current * volts[name])
       
        
        offset = 3.0
       
        return total_internal_power + offset

    except Exception as e:
        print(f"에러: {e}")
        return 0.0


def main():
    print(">>> [전력 모니터링] 시작 (그래프 창이 뜹니다)")
    print(">>> 종료하려면 터미널에서 'Ctrl + C'를 누르세요.")
   
    
    times = []
    powers = []
    start_time = time.time()
   
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
   
    line, = ax.plot([], [], 'b-', linewidth=2, label='Power (W)')
    avg_line = ax.axhline(y=0, color='r', linestyle='--', label='Average')
   
    ax.set_title("Real-time Power Consumption (RPi 5 + Hailo)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 15) 

    try:
        while True:
            
            current_t = time.time() - start_time
            watts = get_real_power()
           
            
            times.append(current_t)
            powers.append(watts)
           
            
            avg_power = np.mean(powers)
           
            
            line.set_data(times, powers)
            avg_line.set_ydata([avg_power]) # 평균선 업데이트
            avg_line.set_label(f'Average: {avg_power:.2f} W')
           
            
            ax.set_xlim(max(0, current_t - 60), current_t + 5) 
            if max(powers) > 14:
                ax.set_ylim(0, max(powers) + 5)
           
            ax.set_title(f"Real-time Power: {watts:.2f} W | Avg: {avg_power:.2f} W")
            ax.legend(loc='upper right')
           
            fig.canvas.draw()
            fig.canvas.flush_events()
           
            
            print(f"Time: {current_t:.1f}s | Power: {watts:.2f} W | Avg: {avg_power:.2f} W")
           
            
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n>>> [측정 종료] 그래프 저장 중...")
       
        
        plt.ioff() # 인터랙티브 모드 끄기
        ax.set_xlim(0, times[-1]) # 전체 시간 범위로 변경
        ax.set_title(f"[Final Result] Avg Power: {np.mean(powers):.2f} W")
       
        
        filename = "power_consumption_graph(2).png"
        plt.savefig(filename)
        print(f">>> 그래프 저장 완료: {filename}")
        print(f">>> 최종 평균 소비 전력: {np.mean(powers):.2f} W")
        plt.close()

if __name__ == "__main__":
    main()
