from .ms_wrap import print
import builtins


def start_analysis():
    print("start_analysis")


def end_analysis():
    print("end_analysis")
    line_index = {}
    with open("performance.log", 'r') as f:
        all_data = f.readlines()
        for index, line in enumerate(all_data):
            if "start_analysis" in line:
                line_index["start_analysis"] = index
            if "end_analysis" in line:
                line_index["end_analysis"] = index
        analysis_time_data = all_data[line_index["start_analysis"]+1:line_index["end_analysis"]]
        result_dc = {}
        for data in analysis_time_data:
            op_name = data.split(" cost_time:")[0].split("INFO: ")[1]
            cost_time = eval(data.split(" cost_time:")[1].strip())
            if op_name not in result_dc:
                # 总耗时，执行此时，最低耗时，最高耗时
                result_dc[op_name] = [cost_time, 1, cost_time, cost_time]
            else:
                result_dc[op_name][0] += cost_time
                result_dc[op_name][1] += 1
                if cost_time < result_dc[op_name][2]:
                    result_dc[op_name][2] = cost_time
                if cost_time > result_dc[op_name][3]:
                    result_dc[op_name][3] = cost_time
        for op_name, op_info in result_dc.items():
            builtins.print(f"{op_name}共执行了{op_info[1]}次,总耗时{op_info[0]}秒,平均耗时{op_info[0]/op_info[1]}秒,最低耗时{op_info[2]},最高耗时{op_info[3]}.")