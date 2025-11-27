import os
import numpy as np
import mne
from scipy import signal
from scipy.signal import iirnotch, filtfilt
import re
from tqdm import tqdm
import json
from datetime import datetime, timezone
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
        
        # 发作数据最小持续时间（秒）
        self.min_seizure_duration = 10  # 只保存持续时间≥10秒的发作片段
        
        # 正常数据最小持续时间（秒）
        self.min_normal_duration = 30  # 只保存持续时间≥30秒的正常数据段

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

            print(f"正在解析summary文件: {summary_file}")
            print(f"文件大小: {len(content)} 字符")
            
            # 预览文件开头
            print(f"文件开头预览: {content[:300].replace('/n', '//n')}...")
            
            # 检查是否有发作相关关键词
            seizure_keywords = ['Seizure Start Time', 'Seizure', '发作']
            for keyword in seizure_keywords:
                count = content.count(keyword)
                print(f"关键词 '{keyword}' 出现次数: {count}")
            
            # 按文件分割 - 尝试多种可能的分隔模式
            file_sections = []
            # 主要分隔模式
            if re.search(r'File Name:\s*', content):
                file_sections = re.split(r'File Name:\s*', content)[1:]
                print(f"使用模式 'File Name:' 找到 {len(file_sections)} 个文件区段")
            
            # 如果没找到足够的区段，尝试其他可能的分隔符
            if len(file_sections) < 2:
                print("尝试备用分隔模式...")
                # 尝试多种可能的分隔模式
                alternative_patterns = [
                    r'\n\s*\d+\s*\n',  # 可能的数字分隔行
                    r'\n{3,}',  # 多个空行
                    r'File Start Time:',  # 文件开始时间
                ]
                
                for pattern in alternative_patterns:
                    test_sections = re.split(pattern, content)
                    if len(test_sections) > len(file_sections):
                        file_sections = test_sections
                        print(f"使用备用模式找到 {len(file_sections)} 个区段")
                        break
            
            print(f"最终处理 {len(file_sections)} 个文件区段")

            for i, section in enumerate(file_sections):
                lines = section.strip().split('\n')
                if not lines:
                    continue

                # 获取文件名 - 更智能地提取
                edf_filename = lines[0].strip()
                # 如果第一行不是有效的文件名，尝试在区段中搜索
                if not (edf_filename.endswith('.edf') or '.edf' in edf_filename):
                    for line in lines:
                        # 尝试在区段中查找包含.edf的行
                        if '.edf' in line:
                            # 提取文件名部分
                            potential_name = re.search(r'(\S+\.edf)', line)
                            if potential_name:
                                edf_filename = potential_name.group(1)
                                print(f"  从行中提取文件名: {edf_filename}")
                                break
                
                print(f"处理文件区段 {i+1}: {edf_filename}")
                
                # 处理文件名 - 确保有.edf后缀
                if not edf_filename.endswith('.edf'):
                    # 如果文件名中已经包含.edf但在中间位置
                    if '.edf' in edf_filename:
                        # 尝试提取正确的文件名
                        match = re.search(r'(\S+\.edf)', edf_filename)
                        if match:
                            edf_filename = match.group(1)
                            print(f"  修正文件名: {edf_filename}")
                    # 如果完全没有.edf后缀，添加它
                    else:
                        edf_filename += '.edf'
                        print(f"  添加.edf后缀: {edf_filename}")

                # 初始化发作信息
                if edf_filename not in seizure_info:
                    seizure_info[edf_filename] = {
                        'seizures': [],
                        'file_start_time': None,
                        'file_end_time': None,
                        'section_index': i
                    }

                # 检查区段中报告的发作数量
                seizure_count = 0
                for line in lines:
                    if 'Number of Seizures' in line:
                        count_match = re.search(r'Number of Seizures[^\d]*?(\d+)', line)
                        if count_match:
                            seizure_count = int(count_match.group(1))
                            print(f"  区段中报告的发作数量: {seizure_count}")
                            break
                
                # 尝试多种模式解析发作信息
                start_time = None
                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    
                    # 尝试多种可能的发作开始时间格式
                    start_patterns = [
                        r'Seizure\s+Start\s+Time:\s*(\d+)\s*(?:seconds|secs?)(?!\s*end)',
                        r'Start\s+Time:\s*(\d+)\s*(?:seconds|secs?)',
                        r'Start:\s*(\d+)\s*(?:seconds|secs?)',
                        r'Seizure\s+\d+\s+Start:\s*(\d+)',
                        r'Seizure\s+starts?\s+at\s*(\d+)\s*(?:seconds|secs?)',
                    ]
                    
                    # 检查是否是发作开始行
                    is_start_line = False
                    for pattern in start_patterns:
                        start_match = re.search(pattern, line, re.IGNORECASE)
                        if start_match:
                            start_time = int(start_match.group(1))
                            print(f"  找到发作开始时间: {start_time}秒 (行 {line_idx+1}: {line})")
                            is_start_line = True
                            break
                    
                    # 如果找到开始时间，尝试查找结束时间
                    if is_start_line and start_time is not None:
                        end_time = None
                        # 先检查当前行后面的几行
                        for j in range(line_idx + 1, min(line_idx + 5, len(lines))):
                            end_line = lines[j].strip()
                            end_patterns = [
                                r'Seizure\s+End\s+Time:\s*(\d+)\s*(?:seconds|secs?)',
                                r'End\s+Time:\s*(\d+)\s*(?:seconds|secs?)',
                                r'End:\s*(\d+)\s*(?:seconds|secs?)',
                                r'Seizure\s+\d+\s+End:\s*(\d+)',
                                r'Seizure\s+ends?\s+at\s*(\d+)\s*(?:seconds|secs?)',
                            ]
                            
                            for end_pattern in end_patterns:
                                end_match = re.search(end_pattern, end_line, re.IGNORECASE)
                                if end_match:
                                    end_time = int(end_match.group(1))
                                    print(f"  找到发作结束时间: {end_time}秒 (行 {j+1}: {end_line})")
                                    break
                            
                            if end_time is not None:
                                break
                        
                        # 如果没找到结束时间，尝试使用开始时间+30秒作为默认值
                        if end_time is None:
                            end_time = start_time + 30
                            print(f"  未找到结束时间，使用默认值: {end_time}秒 (开始时间+30秒)")
                        
                        # 验证发作时间合理性
                        if end_time > start_time:
                            seizure_info[edf_filename]['seizures'].append({
                                'start': start_time,
                                'end': end_time,
                                'source_line': line
                            })
                            print(f"  成功添加发作: {start_time}秒 -> {end_time}秒")
                        else:
                            print(f"  警告: 发作结束时间 ({end_time}) 小于等于开始时间 ({start_time})，跳过")
                        
                        start_time = None
            
            # 打印解析结果
            detected_seizures = sum(len(info['seizures']) for info in seizure_info.values())
            print(f"解析完成，共找到 {len(seizure_info)} 个EDF文件的信息")
            print(f"检测到 {detected_seizures} 个发作")
            
            # 打印有发作的文件
            seizure_files = [f for f, info in seizure_info.items() if info['seizures']]
            if seizure_files:
                print(f"有发作的文件 ({len(seizure_files)}):")
                for f in seizure_files:
                    print(f"  {f}: {len(seizure_info[f]['seizures'])} 个发作")
            else:
                print("未检测到任何发作信息")

        except Exception as e:
            print(f"解析summary文件 {summary_file} 时出错: {e}")
            import traceback
            traceback.print_exc()

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

        # 获取数据总时长（秒）
        total_duration = data.shape[1] / fs
        print(f"  数据总时长: {total_duration:.2f}秒")
        
        # 打印发作信息详情
        print(f"  处理 {len(seizures)} 个发作信息:")
        for i, seizure in enumerate(seizures):
            start_time = seizure.get('start', 0)
            end_time = seizure.get('end', None)
            source_line = seizure.get('source_line', '')
            print(f"    发作 {i+1}: 开始={start_time}秒, 结束={end_time}秒")
            if source_line:
                print(f"      来源: {source_line}")

        for seizure in seizures:
            start_time = seizure.get('start', 0)

            # 处理缺少结束时间的情况
            end_time = seizure.get('end', None)
            if end_time is None:
                print(f"  警告：发作开始于 {start_time} 秒缺少结束时间，尝试使用默认值")
                # 设置默认结束时间为开始时间+30秒
                end_time = start_time + 30
                print(f"  使用默认结束时间: {end_time}秒")

            # 验证时间合理性
            if end_time <= start_time:
                print(f"  警告：发作结束时间 ({end_time}) 早于开始时间 ({start_time})，跳过处理")
                continue

            # 调整超出范围的时间
            if start_time < 0:
                print(f"  警告：发作开始时间 ({start_time}) 小于0，调整为0")
                start_time = 0
            
            if end_time > total_duration:
                print(f"  警告：发作结束时间 ({end_time}) 超出数据范围 ({total_duration:.2f}秒)，调整为数据结束")
                end_time = total_duration
            
            # 再次验证时间有效性
            if start_time >= end_time:
                print(f"  警告：调整后发作时间无效，跳过")
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
                
                # 放宽最小持续时间要求，确保能捕获发作
                if duration < 5:  # 最小5秒
                    print(f"  警告: 发作持续时间过短 ({duration:.2f}秒)，仍然尝试提取")

                seizure_segments.append({
                    'data': seizure_segment,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration
                })

                print(f"  ✓ 成功提取癫痫发作：{start_time}-{end_time} 秒，持续时间 {duration:.1f} 秒")

        print(f"  最终成功提取 {len(seizure_segments)} 个发作片段")
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

    def save_data(self, data, patient_id, file_type, file_index, channel_names=None, sfreq=None, raw_info=None):
        """保存数据为EDF格式"""
        # 创建输出目录
        patient_dir = os.path.join(self.output_root, patient_id)
        type_dir = os.path.join(patient_dir, file_type)
        os.makedirs(type_dir, exist_ok=True)

        # 生成文件名
        if file_type == 'seizure':
            filename = f"S{file_index:03d}.edf"
        else:
            filename = f"N{file_index:03d}.edf"

        filepath = os.path.join(type_dir, filename)

        # 如果没有提供通道名称，创建默认名称
        if channel_names is None:
            num_channels = data.shape[0]
            channel_names = [f'EEG_{i+1}' for i in range(num_channels)]
        
        # 如果没有提供采样率，使用默认值
        if sfreq is None:
            sfreq = self.sample_rate
        
        # 创建MNE的Raw对象来保存EDF
        info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
        
        # 添加更多元数据
        info['subject_info'] = {'id': patient_id}
        info['description'] = f'{file_type.upper()} data for patient {patient_id}, segment {file_index}'
        # 使用正确的方法设置测量日期（UTC格式）
        meas_date = datetime.now(timezone.utc)
        
        # 如果有原始raw对象的信息，可以复制更多元数据
        if raw_info is not None:
            # 复制设备信息（如果有）
            if 'device_info' in raw_info:
                info['device_info'] = raw_info['device_info']
            # 复制通道位置（如果有）
            if 'chs' in raw_info:
                # 确保通道数量匹配
                if len(raw_info['chs']) >= len(channel_names):
                    for i, ch in enumerate(info['chs']):
                        if i < len(raw_info['chs']):
                            ch['loc'] = raw_info['chs'][i]['loc']
        
        # 确保数据类型正确
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(float)
        
        # 创建Raw对象
        raw = mne.io.RawArray(data, info)
        
        # 使用正确的方法设置测量日期
        raw.set_meas_date(meas_date)
        
        # 保存为EDF文件
        raw.export(filepath, fmt='edf', overwrite=True)

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

        # 在患者目录和父目录中查找summary文件（CHB-MIT数据集通常在父目录有summary.txt）
        search_dirs = [patient_dir, os.path.dirname(patient_dir)]
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            
            print(f"在目录 {search_dir} 中查找summary文件...")
            for root, dirs, files in os.walk(search_dir):
                # 只在顶层目录查找，不在子目录中查找
                if root != search_dir:
                    continue
                
                for file in files:
                    print(f"  检查文件: {file}")
                    if any(file.lower() == name.lower() for name in potential_names):
                        summary_path = os.path.join(root, file)
                        summary_files.append(summary_path)
                        print(f"  找到summary文件: {summary_path}")

        if not summary_files:
            print(f"患者 {patient_id}: 未找到summary文件 (搜索了: {', '.join(potential_names)})")
            # 即使没有summary文件，也要继续处理以提取正常数据
            seizure_info = {}
        else:
            # 解析summary文件
            seizure_info = {}
            for summary_file in summary_files:
                print(f"\n解析summary文件: {summary_file}")
                info = self.parse_summary_file(summary_file, patient_id)
                seizure_info.update(info)
            
            # 打印seizure_info的内容以便调试
            print(f"\n总发作信息: {len(seizure_info)} 个文件")
            for edf_file, info in seizure_info.items():
                print(f"  {edf_file}: {len(info['seizures'])} 个发作")

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
                edf_basename = os.path.basename(edf_file)
                print(f"处理文件: {edf_basename}")
                
                # 尝试多种匹配方式查找发作信息
                file_seizures = []
                matched_key = None
                
                # 1. 直接匹配
                if edf_basename in seizure_info:
                    file_seizures = seizure_info[edf_basename].get('seizures', [])
                    matched_key = edf_basename
                    print(f"  通过直接匹配找到发作信息")
                else:
                    # 2. 尝试部分匹配
                    for key in seizure_info.keys():
                        if key in edf_basename or edf_basename in key:
                            file_seizures = seizure_info[key].get('seizures', [])
                            matched_key = key
                            print(f"  通过部分匹配找到发作信息: {key} -> {edf_basename}")
                            break
                    
                    # 3. 如果还是没找到，尝试去除扩展名匹配
                    if not matched_key:
                        edf_no_ext = os.path.splitext(edf_basename)[0]
                        for key in seizure_info.keys():
                            key_no_ext = os.path.splitext(key)[0]
                            if key_no_ext == edf_no_ext:
                                file_seizures = seizure_info[key].get('seizures', [])
                                matched_key = key
                                print(f"  通过无扩展名匹配找到发作信息: {key} -> {edf_basename}")
                                break
                    
                    # 4. 如果还是没找到，尝试数字部分匹配
                    if not matched_key:
                        # 提取文件名中的数字部分
                        edf_numbers = re.findall(r'\d+', edf_basename)
                        if edf_numbers:
                            for key in seizure_info.keys():
                                key_numbers = re.findall(r'\d+', key)
                                # 检查是否有数字重叠
                                if any(num in key_numbers for num in edf_numbers):
                                    # 计算匹配的数字数量
                                    match_count = sum(1 for num in edf_numbers if num in key_numbers)
                                    # 只有当至少有两个数字匹配或文件名相似度高时才使用
                                    if match_count >= 2 or len(edf_no_ext) > 0 and edf_no_ext in key:
                                        file_seizures = seizure_info[key].get('seizures', [])
                                        matched_key = key
                                        print(f"  通过数字部分匹配找到发作信息: {key} -> {edf_basename} (匹配数字: {match_count})")
                                        break
                
                print(f"  该文件发作信息: {len(file_seizures)} 个发作")
                
                # 打印找到的具体发作信息
                for i, seizure in enumerate(file_seizures):
                    print(f"    发作 {i+1}: {seizure['start']}秒 -> {seizure['end']}秒")
                    # 如果有来源行信息，也打印出来
                    if 'source_line' in seizure:
                        print(f"      来源行: {seizure['source_line']}")
                
                # 如果没找到发作信息，提供额外的调试信息
                if len(file_seizures) == 0:
                    print(f"  未找到该文件的发作信息")
                    # 打印所有可用的发作信息键，以便调试
                    if len(seizure_info) > 0:
                        print(f"  可用的发作信息键 (前5个): {list(seizure_info.keys())[:5]}")
                        # 尝试计算文件名相似度
                        similarity_scores = []
                        edf_no_ext = os.path.splitext(edf_basename)[0]
                        for key in seizure_info.keys():
                            key_no_ext = os.path.splitext(key)[0]
                            # 简单的相似度计算：共同字符数
                            common_chars = len(set(edf_no_ext) & set(key_no_ext))
                            similarity = common_chars / max(len(edf_no_ext), len(key_no_ext))
                            similarity_scores.append((key, similarity))
                        # 排序并显示最相似的几个
                        similarity_scores.sort(key=lambda x: x[1], reverse=True)
                        print(f"  最相似的发作信息键 (前3个):")
                        for key, score in similarity_scores[:3]:
                            print(f"    {key}: {score:.2f} 相似度")

                # 提取癫痫发作数据
                if file_seizures:
                    seizure_segments = self.extract_seizure_data(data, file_seizures, fs)

                    for i, segment in enumerate(seizure_segments):
                        if segment['duration'] >= 10:  # 只保存持续时间>=10秒的发作
                            self.save_data(
                                segment['data'],
                                patient_id,
                                'seizure',
                                seizure_count + 1,
                                channel_names=raw.ch_names,
                                sfreq=fs,
                                raw_info=raw.info
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
                            normal_count + 1,
                            channel_names=raw.ch_names,
                            sfreq=fs,
                            raw_info=raw.info
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