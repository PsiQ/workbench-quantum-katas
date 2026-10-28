# Draw a histogram for alias sampling explanations in a style similar to https://algassert.com/post/1805
# and save it in an svg file.

import svgwrite

colors = ['#532098', '#FFFFFF', '#8F3FFC', '#D0B0FF']
box_size = 20

for filename, heights, aliases, thresholds in [
    ('initial', [7, 3, 1, 5], [0, 0, 0, 0], [9, 9, 9, 9]),
    ('repack1', [4, 4, 4, 4], [0, 0, 3, 0], [4, 3, 1, 2]),
    ('repack2', [4, 4, 4, 4], [0, 3, 0, 0], [4, 3, 1, 4]),
    ('repack1_step1', [6, 4, 1, 5], [0, 0, 0, 0], [9, 3, 9, 9]),
    ('repack1_step2', [6, 4, 4, 2], [0, 0, 3, 0], [9, 3, 1, 9]),

]:
    dwg = svgwrite.Drawing(f'{filename}.svg')

    bottom = max(heights)
    for i in range(len(heights)):
        for j in range(heights[i]):
            color_ind = i if j < thresholds[i] else aliases[i]
            dwg.add(dwg.rect(insert=(i * 2 * box_size, (bottom - j - 1) * box_size), size=(box_size, box_size), 
                            fill=colors[color_ind], stroke='black', stroke_width=1))
            dwg.add(dwg.text(f'{color_ind}', insert=(i * 2 * box_size + (box_size / 2 - 3), 
                                                     (bottom - j - 1) * box_size + box_size / 2 + 4), 
                            font_size='10px', font_family="Arial", fill='white' if color_ind % 2 == 0 else 'black'))

    dwg.viewbox(-box_size, -box_size, box_size * (2 * len(heights) + 1), box_size * (max(heights) + 2))
    dwg.save()
