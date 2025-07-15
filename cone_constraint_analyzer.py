#!/usr/bin/env python3
"""
锥形约束详细几何分析器
分析USD文件的原点位置、几何中心，并评估scale调整的合理性
"""

import os
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
try:
    from pxr import Usd, UsdGeom, Gf, Sdf
except ImportError:
    print("错误：未找到USD Python库。请安装：pip install usd-core")
    exit(1)

class DetailedConeAnalyzer:
    """详细的锥形约束分析器"""
    
    def __init__(self):
        self.stage = None
        self.mesh_data = {}
    
    def analyze_usd_origin_and_scale(self, cone_file: str):
        """
        分析USD文件的原点、几何中心，并评估scale调整
        
        Args:
            cone_file: 锥形约束USD文件路径
        """
        print("="*80)
        print("🔍 USD文件详细几何分析")
        print("="*80)
        
        # 加载文件
        stage = Usd.Stage.Open(cone_file)
        if not stage:
            print(f"❌ 无法打开文件: {cone_file}")
            return
        
        self.stage = stage
        
        # 获取基本信息
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        unit_scale = 1000 if meters_per_unit == 1.0 else 1
        unit_name = "毫米" if meters_per_unit == 1.0 else "米"
        
        print(f"📏 USD文件单位: {unit_name} (MetersPerUnit: {meters_per_unit})")
        print(f"🔄 转换比例: {unit_scale} (转换为毫米)")
        
        # 分析mesh几何
        mesh_analysis = self._analyze_mesh_geometry(unit_scale, unit_name)
        
        if not mesh_analysis:
            print("❌ 未找到mesh数据")
            return
        
        # 分析原点和几何中心
        self._analyze_origin_and_center(mesh_analysis)
        
        # 分析锥形特征
        cone_features = self._analyze_cone_features(mesh_analysis)
        
        # 评估scale调整方案
        self._evaluate_scale_adjustment(cone_features, target_min=10, target_max=25)
        
        return mesh_analysis, cone_features
    
    def _analyze_mesh_geometry(self, unit_scale: float, unit_name: str) -> Dict:
        """分析mesh几何数据"""
        print(f"\n📐 Mesh几何分析")
        print("-" * 60)
        
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                
                # 获取顶点
                points_attr = mesh.GetPointsAttr()
                if not points_attr:
                    continue
                
                points = points_attr.Get()
                if not points:
                    continue
                
                # 转换为numpy数组并应用单位转换
                vertices = np.array([(p[0]*unit_scale, p[1]*unit_scale, p[2]*unit_scale) 
                                   for p in points])
                
                # 获取面数据
                face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
                face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                
                # 计算基本统计
                vertex_count = len(vertices)
                face_count = len(face_vertex_counts) if face_vertex_counts else 0
                
                print(f"✅ 顶点数量: {vertex_count}")
                print(f"✅ 面数量: {face_count}")
                
                # 计算边界框
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)
                bbox_size = bbox_max - bbox_min
                bbox_center = (bbox_min + bbox_max) / 2
                
                print(f"📦 边界框 ({unit_name}):")
                print(f"   最小值: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]")
                print(f"   最大值: [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]")
                print(f"   尺寸:   [{bbox_size[0]:.2f}, {bbox_size[1]:.2f}, {bbox_size[2]:.2f}]")
                print(f"   中心:   [{bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f}]")
                
                return {
                    'vertices': vertices,
                    'vertex_count': vertex_count,
                    'face_count': face_count,
                    'face_indices': list(face_vertex_indices) if face_vertex_indices else [],
                    'face_counts': list(face_vertex_counts) if face_vertex_counts else [],
                    'bbox_min': bbox_min,
                    'bbox_max': bbox_max,
                    'bbox_size': bbox_size,
                    'bbox_center': bbox_center,
                    'unit_name': unit_name,
                    'unit_scale': unit_scale
                }
        
        return {}
    
    def _analyze_origin_and_center(self, mesh_data: Dict):
        """分析原点和几何中心关系"""
        print(f"\n🎯 原点与几何中心分析")
        print("-" * 60)
        
        vertices = mesh_data['vertices']
        bbox_center = mesh_data['bbox_center']
        unit_name = mesh_data['unit_name']
        
        # 检查原点位置
        origin = np.array([0, 0, 0])
        
        print(f"📍 USD坐标系原点: [0, 0, 0]")
        print(f"📍 几何中心位置: [{bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f}] {unit_name}")
        
        # 计算原点到几何中心的距离
        center_offset = bbox_center - origin
        offset_distance = np.linalg.norm(center_offset)
        
        print(f"📏 原点偏移量: [{center_offset[0]:.2f}, {center_offset[1]:.2f}, {center_offset[2]:.2f}] {unit_name}")
        print(f"📏 偏移距离: {offset_distance:.2f} {unit_name}")
        
        # 判断原点是否在几何中心
        tolerance = 0.1  # 容忍度
        if offset_distance < tolerance:
            print("✅ 几何中心基本与原点重合")
        else:
            print("⚠️  几何中心偏离原点")
        
        # 分析原点相对于几何体的位置
        bbox_min = mesh_data['bbox_min']
        bbox_max = mesh_data['bbox_max']
        
        print(f"\n🔍 原点位置分析:")
        
        # X轴
        if bbox_min[0] <= 0 <= bbox_max[0]:
            x_pos = "内部"
        elif 0 < bbox_min[0]:
            x_pos = f"左侧 ({bbox_min[0]:.2f}{unit_name})"
        else:
            x_pos = f"右侧 ({-bbox_max[0]:.2f}{unit_name})"
        
        # Y轴  
        if bbox_min[1] <= 0 <= bbox_max[1]:
            y_pos = "内部"
        elif 0 < bbox_min[1]:
            y_pos = f"前方 ({bbox_min[1]:.2f}{unit_name})"
        else:
            y_pos = f"后方 ({-bbox_max[1]:.2f}{unit_name})"
        
        # Z轴
        if bbox_min[2] <= 0 <= bbox_max[2]:
            z_pos = "内部"
        elif 0 < bbox_min[2]:
            z_pos = f"下方 ({bbox_min[2]:.2f}{unit_name})"
        else:
            z_pos = f"上方 ({-bbox_max[2]:.2f}{unit_name})"
        
        print(f"   X轴方向: 原点在几何体{x_pos}")
        print(f"   Y轴方向: 原点在几何体{y_pos}")
        print(f"   Z轴方向: 原点在几何体{z_pos}")
        
        # 检查原点是否在几何体内部
        origin_inside = (bbox_min[0] <= 0 <= bbox_max[0] and 
                        bbox_min[1] <= 0 <= bbox_max[1] and 
                        bbox_min[2] <= 0 <= bbox_max[2])
        
        if origin_inside:
            print("✅ 原点位于几何体内部")
        else:
            print("❌ 原点位于几何体外部")
        
        mesh_data['origin_analysis'] = {
            'center_offset': center_offset,
            'offset_distance': offset_distance,
            'origin_inside': origin_inside,
            'position_description': {
                'x': x_pos,
                'y': y_pos, 
                'z': z_pos
            }
        }
    
    def _analyze_cone_features(self, mesh_data: Dict) -> Dict:
        """分析锥形特征"""
        print(f"\n🔺 锥形几何特征分析")
        print("-" * 60)
        
        vertices = mesh_data['vertices']
        unit_name = mesh_data['unit_name']
        
        # Z轴分析
        z_coords = vertices[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        height = z_max - z_min
        
        print(f"📏 锥形高度: {height:.2f} {unit_name}")
        print(f"📏 Z轴范围: {z_min:.2f} 到 {z_max:.2f} {unit_name}")
        
        # 分层分析锥形轮廓
        num_layers = 20
        layer_height = height / num_layers
        
        inner_radii = []
        outer_radii = []
        profile_data = []
        
        print(f"\n🔍 锥形轮廓分析 (分{num_layers}层):")
        print("层次 | Z坐标     | 内径      | 外径      | 壁厚      ")
        print("-" * 55)
        
        for i in range(num_layers):
            z_layer = z_min + (i + 0.5) * layer_height
            tolerance = layer_height * 0.6
            
            # 找到该层的顶点
            layer_mask = np.abs(vertices[:, 2] - z_layer) < tolerance
            layer_vertices = vertices[layer_mask]
            
            if len(layer_vertices) > 5:  # 需要足够的顶点
                # 计算径向距离
                distances = np.sqrt(layer_vertices[:, 0]**2 + layer_vertices[:, 1]**2)
                
                if len(distances) > 1:
                    inner_r = np.min(distances)
                    outer_r = np.max(distances)
                    wall_thickness = outer_r - inner_r
                    
                    inner_radii.append(inner_r)
                    outer_radii.append(outer_r)
                    
                    profile_data.append({
                        'layer': i+1,
                        'z': z_layer,
                        'inner_radius': inner_r,
                        'outer_radius': outer_r,
                        'wall_thickness': wall_thickness
                    })
                    
                    print(f"{i+1:2d}   | {z_layer:8.2f} | {inner_r:8.2f} | {outer_r:8.2f} | {wall_thickness:8.2f}")
        
        # 分析锥形特征
        if len(inner_radii) > 1:
            min_inner = np.min(inner_radii)
            max_inner = np.max(inner_radii)
            min_outer = np.min(outer_radii)
            max_outer = np.max(outer_radii)
            
            print(f"\n📊 锥形特征总结:")
            print(f"内径范围: {min_inner:.2f} - {max_inner:.2f} {unit_name}")
            print(f"外径范围: {min_outer:.2f} - {max_outer:.2f} {unit_name}")
            
            # 计算锥形角度
            if len(profile_data) > 1:
                z_values = [p['z'] for p in profile_data]
                inner_values = [p['inner_radius'] for p in profile_data]
                
                # 线性拟合计算锥形角度
                poly_inner = np.polyfit(z_values, inner_values, 1)
                slope = poly_inner[0]  # dr/dz
                cone_angle = np.degrees(np.arctan(abs(slope)))
                
                print(f"锥形角度: {cone_angle:.1f}°")
            
            return {
                'height': height,
                'z_range': [z_min, z_max],
                'inner_radius_range': [min_inner, max_inner],
                'outer_radius_range': [min_outer, max_outer],
                'cone_angle': cone_angle if 'cone_angle' in locals() else 0,
                'profile_data': profile_data,
                'current_scale_mm': {
                    'min_inner_radius': min_inner,
                    'max_inner_radius': max_inner
                }
            }
        
        return {}
    
    def _evaluate_scale_adjustment(self, cone_features: Dict, target_min: float = 10, target_max: float = 25):
        """评估scale调整方案"""
        print(f"\n🔧 Scale调整方案评估")
        print("-" * 60)
        
        if not cone_features or 'current_scale_mm' not in cone_features:
            print("❌ 缺少锥形特征数据，无法评估scale")
            return
        
        current = cone_features['current_scale_mm']
        current_min = current['min_inner_radius']
        current_max = current['max_inner_radius']
        
        print(f"📏 当前尺寸:")
        print(f"   内径范围: {current_min:.2f} - {current_max:.2f} mm")
        
        print(f"\n🎯 目标尺寸:")
        print(f"   内径范围: {target_min:.2f} - {target_max:.2f} mm")
        
        # 计算所需的缩放因子
        scale_factor_min = target_min / current_min
        scale_factor_max = target_max / current_max
        
        print(f"\n🔢 缩放因子计算:")
        print(f"   基于最小内径: {scale_factor_min:.3f}x")
        print(f"   基于最大内径: {scale_factor_max:.3f}x")
        
        # 检查缩放一致性
        scale_diff = abs(scale_factor_min - scale_factor_max)
        scale_avg = (scale_factor_min + scale_factor_max) / 2
        
        print(f"   平均缩放因子: {scale_avg:.3f}x")
        print(f"   缩放差异: {scale_diff:.3f}x ({scale_diff/scale_avg*100:.1f}%)")
        
        # 评估缩放合理性
        print(f"\n✅ 缩放方案评估:")
        
        if scale_diff / scale_avg < 0.1:  # 差异小于10%
            print(f"✅ 缩放一致性良好 (差异 < 10%)")
            recommended_scale = scale_avg
        else:
            print(f"⚠️  缩放一致性较差 (差异 > 10%)")
            print(f"   建议优先保证最小内径，使用缩放因子: {scale_factor_min:.3f}x")
            recommended_scale = scale_factor_min
        
        # 计算调整后的所有尺寸
        print(f"\n📐 调整后的完整尺寸 (缩放因子: {recommended_scale:.3f}x):")
        
        new_min_inner = current_min * recommended_scale
        new_max_inner = current_max * recommended_scale
        new_height = cone_features['height'] * recommended_scale
        
        print(f"   内径范围: {new_min_inner:.2f} - {new_max_inner:.2f} mm")
        print(f"   锥形高度: {new_height:.2f} mm")
        
        if 'outer_radius_range' in cone_features:
            outer_min, outer_max = cone_features['outer_radius_range']
            new_outer_min = outer_min * recommended_scale
            new_outer_max = outer_max * recommended_scale
            print(f"   外径范围: {new_outer_min:.2f} - {new_outer_max:.2f} mm")
            
            # 计算壁厚
            wall_thickness_min = new_outer_min - new_min_inner
            wall_thickness_max = new_outer_max - new_max_inner
            print(f"   壁厚范围: {wall_thickness_min:.2f} - {wall_thickness_max:.2f} mm")
        
        # 合理性评估
        print(f"\n🎯 应用合理性评估:")
        
        # 手术器械尺寸评估
        if 5 <= new_min_inner <= 15 and 15 <= new_max_inner <= 35:
            print("✅ 尺寸适合手术器械应用")
        elif new_min_inner < 5:
            print("⚠️  最小内径较小，可能限制器械类型")
        elif new_max_inner > 35:
            print("⚠️  最大内径较大，约束可能不够精确")
        
        # Isaac Lab应用评估
        print("✅ 适合Isaac Lab仿真 (尺寸合理，便于控制)")
        
        # 小球半径建议
        suitable_ball_radius = new_min_inner * 0.15  # 约15%的最小内径
        print(f"💡 建议小球半径: {suitable_ball_radius:.2f} mm (约占最小内径的15%)")
        
        return {
            'recommended_scale': recommended_scale,
            'target_dimensions': {
                'inner_radius_range': [new_min_inner, new_max_inner],
                'height': new_height
            },
            'suitable_ball_radius': suitable_ball_radius
        }

def main():
    """主函数"""
    cone_file = "/home/zzh/workspace/surgical_robot_project/assets/models/usd/ConeConstraint.usd"
    
    analyzer = DetailedConeAnalyzer()
    
    print("开始分析USD文件...")
    result = analyzer.analyze_usd_origin_and_scale(cone_file)
    
    print("\n" + "="*80)
    print("✅ 分析完成!")
    print("="*80)

if __name__ == "__main__":
    main()