import os
import numpy as np
import mne
from scipy import signal
from scipy.signal import iirnotch, filtfilt
import re
from tqdm import tqdm
import json
from datetime import datetime
import warnings



class CHBMITPreprocessor:
    def __init__(self, data_root, output_root, sample_rate=256):
        """
        CHB-MIT数据集预处理器
        
        参数:
        - data_root: CHB-MIT数据集根目录
        - output_root: 输出目录
        - sample_rate: 目标采样率
        """
        self.data_root = data_root
        self.output_root = output_root
        self.sample_rate = sample_rate
        
        # 60Hz工频陷波滤波器参数
        self.notch_freq = 60.0
        self.notch_quality = 30.0
        
        # 带通滤波器参数 (0.5-70 Hz)
        self.low_freq = 0.5
        self.high_freq = 70.0
        
        # 发作前后去除时间 (5分钟 = 300秒)
        self.exclude_time = 300  # 秒
        
    def find_summary_files(self):
        """查找所有summary.txt文件"""
        summary_files = []
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.lower() == 'summary.txt':
                    summary_files.append(os.path.join(root, file))
        return summary_files
    
    def find_edf_files(self, patient_dir):
        """查找患者目录下的所有EDF文件"""
        edf_files = []
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if file.lower().endswith('.edf'):
                    edf_files.append(os.path.join(root, file))
        return sorted(edf_files)
    
    def parse_summary_file(self, summary_file, patient_id):
        """解析summary.txt文件，提取癫痫发作信息"""
        seizure_info = {}
        
        try:
            with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 按文件分割
            file_sections = re.split(r'File Name:\s*', content)[1:]
            
            for section in file_sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                
                # 获取文件名
                edf_filename = lines[0].strip()
                if not edf_filename.endswith('.edf'):
                    continue
                
                seizure_info[edf_filename] = {
                    'seizures': [],
                    'file_start_time': None,
                    'file_end_time': None
                }
                
                # 解析发作信息
                current_line = 1
                while current_line < len(lines):
                    line = lines[current_line].strip()
                    
                    # 查找发作开始时间
                    if 'Seizure Start Time:' in line:
                        start_match = re.search(r'Seizure Start Time:\s*(\d+)\s*seconds', line)
                        if start_match:
                            start_time = int(start_match.group(1))
                            
                            # 查找发作结束时间
                            end_time = None
                            if current_line + 1 < len(lines) and 'Seizure End Time:' in lines[current_line + 1]:
                                end_match = re.search(r'Seizure End Time:\s*(\d+)\s*seconds', lines[current_line + 1])
                                if end_match:
                                    end_time = int(end_match.group(1))
                            
                            seizure_info[edf_filename]['seizures'].append({
                                'start': start_time,
                                'end': end_time
                            })
                    
                    current_line += 1
                
        except Exception as e:
            print(f"解析summary文件 {summary_file} 时出错: {e}")
        
        return seizure_info
    
    def detect_bad_channels(self, data, channel_names, fs):
        """
        检测坏道 - 简化为只检测全零通道
        
        参数:
        - data: EEG数据，形状为(n_channels, n_samples)
        - channel_names: 通道名称列表
        - fs: 采样率
        
        返回:
        - bad_channels: 检测到的坏道列表（仅包含全零通道）
        """
        bad_channels = []
        n_samples = data.shape[1]
        
        for i, (ch_name, ch_data) in enumerate(zip(channel_names, data)):
            # 只检查是否全为0（或接近0）
            zero_ratio = np.sum(np.abs(ch_data) < 1e-10) / n_samples
            
            if zero_ratio > 0.99:  # 99%以上的值为0，认为是坏道
                bad_channels.append(ch_name)
                print(f"  通道 {ch_name}: 检测到坏道 - 全零数据({zero_ratio:.2f})")
        
        return bad_channels
    
    def remove_bad_channels(self, raw, patient_id):
        """去除坏道 - 基于数据质量检测（使用mne）"""
        # 获取数据和通道信息
        data = raw.get_data()
        channel_names = raw.ch_names
        fs = raw.info['sfreq']
        
        # 检测坏道
        bad_chs = self.detect_bad_channels(data, channel_names, fs)
        
        if bad_chs:
            print(f"患者 {patient_id}: 检测到 {len(bad_chs)} 个坏道: {bad_chs}")
            raw.info['bads'] = bad_chs
            
            try:
                raw.interpolate_bads()  # 使用插值填充坏道
                print(f"患者 {patient_id}: 已去除并插值坏道")
            except Exception as e:
                print(f"患者 {patient_id}: 插值失败，将直接丢弃坏道: {e}")
                # 如果插值失败，直接删除坏道
                good_channels = [ch for ch in channel_names if ch not in bad_chs]
                raw.pick_channels(good_channels)
        
        return raw
    
    def apply_notch_filter(self, data, fs):
        """应用60Hz陷波滤波器"""
        # 设计陷波滤波器
        w0 = self.notch_freq / (fs / 2)
        b, a = iirnotch(w0, self.notch_quality)
        
        # 应用滤波器
        filtered_data = filtfilt(b, a, data, axis=1)
        
        return filtered_data
    
    def apply_bandpass_filter(self, data, fs):
        """应用0.5-70Hz带通滤波器"""
        # 设计巴特沃斯滤波器
        low = self.low_freq / (fs / 2)
        high = self.high_freq / (fs / 2)
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 应用滤波器
        filtered_data = filtfilt(b, a, data, axis=1)
        
        return filtered_data
    
    def extract_seizure_data(self, data, seizures, fs):
        """提取癫痫发作数据段"""
        seizure_segments = []
        
        for seizure in seizures:
            start_time = seizure['start']
            
            # 如果缺少结束时间，记录警告并跳过该发作
            if seizure['end'] is None:
                print(f"  ⚠️  警告：发作开始于 {start_time} 秒缺少结束时间，跳过处理")
                continue
            
            end_time = seizure['end']
            
            # 验证时间合理性
            if end_time <= start_time:
                print(f"  ⚠️  警告：发作结束时间 ({end_time}) 早于开始时间 ({start_time})，跳过处理")
                continue
            
            # 转换为采样点
            start_sample = int(start_time * fs)
            end_sample = int(end_time * fs)
            
            # 确保不超出数据范围
            start_sample = max(0, start_sample)
            end_sample = min(data.shape[1], end_sample)
            
            if end_sample > start_sample:
                seizure_segment = data[:, start_sample:end_sample]
                duration = (end_sample - start_sample) / fs
                
                seizure_segments.append({
                    'data': seizure_segment,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })
                
                print(f"  提取癫痫发作：{start_time}-{end_time} 秒，持续时间 {duration:.1f} 秒")
        
        return seizure_segments
    
    def extract_normal_data(self, data, seizures, fs, file_duration):
        """提取正常数据段（去除发作前后5分钟）"""
        normal_segments = []
        
        # 创建排除区间
        exclude_intervals = []
        for seizure in seizures:
            start_time = seizure['start']
            
            # 如果缺少结束时间，使用更保守的排除策略
            if seizure['end'] is None:
                print(f"  ⚠️  警告：发作开始于 {start_time} 秒缺少结束时间，使用保守排除策略")
                # 只排除发作开始前后各5分钟，不假设持续时间
                exclude_start = max(0, start_time - self.exclude_time)
                exclude_end = min(file_duration, start_time + self.exclude_time)
            else:
                end_time = seizure['end']
                if end_time <= start_time:
                    print(f"  ⚠️  警告：发作结束时间 ({end_time}) 早于开始时间 ({start_time})，跳过该发作的排除")
                    continue
                exclude_start = max(0, start_time - self.exclude_time)
                exclude_end = min(file_duration, end_time + self.exclude_time)
            
            exclude_intervals.append((exclude_start, exclude_end))
        
        # 合并重叠的排除区间
        exclude_intervals = self.merge_intervals(exclude_intervals)
        
        # 找到正常数据区间
        normal_intervals = []
        current_time = 0
        
        for start, end in exclude_intervals:
            if current_time < start:
                normal_intervals.append((current_time, start))
            current_time = max(current_time, end)
        
        if current_time < file_duration:
            normal_intervals.append((current_time, file_duration))
        
        # 提取正常数据
        for start_time, end_time in normal_intervals:
            start_sample = int(start_time * fs)
            end_sample = int(end_time * fs)
            
            if end_sample > start_sample:
                normal_segment = data[:, start_sample:end_sample]
                normal_segments.append({
                    'data': normal_segment,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': (end_sample - start_sample) / fs
                })
        
        return normal_segments
    
    def merge_intervals(self, intervals):
        """合并重叠的时间区间"""
        if not intervals:
            return []
        
        # 按开始时间排序
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # 有重叠
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def save_data(self, data, patient_id, file_type, file_index):
        """保存数据为npy格式"""
        # 创建输出目录
        patient_dir = os.path.join(self.output_root, patient_id)
        type_dir = os.path.join(patient_dir, file_type)
        os.makedirs(type_dir, exist_ok=True)
        
        # 生成文件名
        if file_type == 'seizure':
            filename = f"S{file_index:03d}.npy"
        else:
            filename = f"N{file_index:03d}.npy"
        
        filepath = os.path.join(type_dir, filename)
        
        # 保存数据
        np.save(filepath, data)
        
        return filepath
    
    def process_patient(self, patient_dir):
        """处理单个患者的数据"""
        patient_id = os.path.basename(patient_dir)
        print(f"\n处理患者 {patient_id}...")
        
        # 查找summary文件 - 支持多种命名格式
        summary_files = []
        potential_names = [
            'summary.txt',
            f'{patient_id}-summary.txt',
            f'chb{patient_id[3:]}-summary.txt',  # 提取数字部分，如chb01->01
            f'{patient_id}_summary.txt',
            f'chb{patient_id[3:]}_summary.txt',
        ]
        
        # 在患者目录中查找summary文件
        for root, dirs, files in os.walk(patient_dir):
            for file in files:
                if any(file.lower() == name.lower() for name in potential_names):
                    summary_files.append(os.path.join(root, file))
        
        if not summary_files:
            print(f"患者 {patient_id}: 未找到summary文件 (搜索了: {', '.join(potential_names)})")
            return
        
        # 解析summary文件
        seizure_info = {}
        for summary_file in summary_files:
            info = self.parse_summary_file(summary_file, patient_id)
            seizure_info.update(info)
        
        # 查找EDF文件
        edf_files = self.find_edf_files(patient_dir)
        
        seizure_count = 0
        normal_count = 0
        
        for edf_file in tqdm(edf_files, desc=f"处理 {patient_id}"):
            try:
                # 抑制通道名称重复的警告
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='Channel names are not unique')
                
                # 读取EDF文件
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                
                # 检查并处理重复的通道名称
                channel_names = raw.ch_names
                unique_names = []
                seen_names = set()
                
                for name in channel_names:
                    if name in seen_names:
                        # 如果名称已存在，添加后缀使其唯一
                        counter = 1
                        new_name = f"{name}_{counter}"
                        while new_name in seen_names:
                            counter += 1
                            new_name = f"{name}_{counter}"
                        unique_names.append(new_name)
                        seen_names.add(new_name)
                        print(f"  警告: 通道 '{name}' 重复，重命名为 '{new_name}'")
                    else:
                        unique_names.append(name)
                        seen_names.add(name)
                
                # 如果有重命名，更新通道名称
                if unique_names != channel_names:
                    raw.rename_channels(dict(zip(channel_names, unique_names)))
                
                # 获取文件名
                edf_filename = os.path.basename(edf_file)
                
                # 去除坏道
                raw = self.remove_bad_channels(raw, patient_id)
                
                # 获取数据
                data = raw.get_data()
                fs = raw.info['sfreq']
                file_duration = data.shape[1] / fs
                
                # 重采样到目标采样率
                if fs != self.sample_rate:
                    data = signal.resample(data, int(data.shape[1] * self.sample_rate / fs), axis=1)
                    fs = self.sample_rate
                
                # 应用60Hz陷波滤波器
                data = self.apply_notch_filter(data, fs)
                
                # 应用0.5-70Hz带通滤波器
                data = self.apply_bandpass_filter(data, fs)
                
                # 获取该文件的发作信息
                file_seizures = seizure_info.get(edf_filename, {}).get('seizures', [])
                
                # 提取癫痫发作数据
                if file_seizures:
                    seizure_segments = self.extract_seizure_data(data, file_seizures, fs)
                    
                    for i, segment in enumerate(seizure_segments):
                        if segment['duration'] >= 10:  # 只保存持续时间>=10秒的发作
                            self.save_data(
                            segment['data'], 
                            patient_id, 
                            'seizure', 
                            seizure_count + 1
                        )
                            seizure_count += 1
                            print(f"  保存癫痫发作数据: S{seizure_count:03d} ({segment['duration']:.1f}秒)")
        
        # 提取正常数据
                normal_segments = self.extract_normal_data(data, file_seizures, fs, file_duration)
                
                for i, segment in enumerate(normal_segments):
                    if segment['duration'] >= 30:  # 只保存持续时间>=30秒的正常数据
                        # 保存完整的正常数据段，不进行切割
                        normal_data = segment['data']
                        self.save_data(
                            normal_data, 
                            patient_id, 
                            'normal', 
                            normal_count + 1
                        )
                        normal_count += 1
                
            except Exception as e:
                print(f"处理文件 {edf_file} 时出错: {e}")
                continue
        
        print(f"患者 {patient_id}: 处理完成")
        print(f"  癫痫发作片段: {seizure_count} 个")
        print(f"  正常数据段: {normal_count} 段")
        
        return {
            'patient_id': patient_id,
            'seizure_count': seizure_count,
            'normal_count': normal_count
        }
    
    def process_all_patients(self):
        """处理所有患者的数据"""
        print("开始CHB-MIT数据集预处理...")
        print(f"数据根目录: {self.data_root}")
        print(f"输出目录: {self.output_root}")
        
        # 创建输出目录
        os.makedirs(self.output_root, exist_ok=True)
        
        # 查找所有患者目录
        patient_dirs = []
        
        # 检查当前data_root是否本身就是一个患者目录
        if os.path.basename(self.data_root).lower().startswith('chb'):
            # 如果是单个患者目录，直接处理
            patient_dirs.append(self.data_root)
            print(f"检测到单个患者目录: {os.path.basename(self.data_root)}")
        else:
            # 否则查找子目录中的患者目录
            for item in os.listdir(self.data_root):
                item_path = os.path.join(self.data_root, item)
                if os.path.isdir(item_path) and item.lower().startswith('chb'):
                    patient_dirs.append(item_path)
        
        if not patient_dirs:
            print("未找到患者目录（目录名应以'chb'开头）")
            return
        
        print(f"找到 {len(patient_dirs)} 个患者目录")
        
        # 处理每个患者
        results = []
        for patient_dir in sorted(patient_dirs):
            try:
                result = self.process_patient(patient_dir)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"处理患者 {os.path.basename(patient_dir)} 时出错: {e}")
                continue
        
        # 保存处理统计信息
        self.save_statistics(results)
        
        print("\n预处理完成！")
        print(f"总计处理了 {len(results)} 个患者")
        
        total_seizures = sum(r['seizure_count'] for r in results)
        total_normal = sum(r['normal_count'] for r in results)
        
        print(f"总癫痫发作片段: {total_seizures} 个")
        print(f"总正常数据段: {total_normal} 段")
    
    def save_statistics(self, results):
        """保存处理统计信息"""
        stats = {
            'processing_time': datetime.now().isoformat(),
            'total_patients': len(results),
            'total_seizures': sum(r['seizure_count'] for r in results),
            'total_normal_segments': sum(r['normal_count'] for r in results),
            'patient_details': results
        }
        
        stats_file = os.path.join(self.output_root, 'processing_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"统计信息已保存到: {stats_file}")


def main():
    """主函数 - 支持单个患者目录或患者集合目录"""
    # 配置路径 - 直接在这里设置你的数据路径
    data_root = r"D:\eeg癫痫模型\sjyz\data\archive\chb-mit-scalp-eeg-database-1.0.0\chb06"  # 可以是单个患者目录或多个患者目录的父目录
    output_root = r"C:\Users\admin\Desktop\新建文件夹 (3)\1"  # 修改为你的输出路径
    
    # 验证数据路径
    if not os.path.exists(data_root):
        print(f"错误: 数据目录 {data_root} 不存在")
        print("请修改main函数中的data_root路径为你的CHB-MIT数据集路径")
        return
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    print(f"数据根目录: {data_root}")
    print(f"输出目录: {output_root}")
    
    # 创建预处理器
    preprocessor = CHBMITPreprocessor(
        data_root=data_root,
        output_root=output_root,
        sample_rate=256  # 采样率
    )
    
    # 开始处理
    preprocessor.process_all_patients()


if __name__ == "__main__":
    main()