def unscramble(trace):
    """ script to fix scrambled traces (Note: first and last 64 samples are unusable and hence masked with zeros) """
    new_trace = np.zeros_like(trace)
    for i_section in range(len(trace) // 64):
        section_start = i_section * 64
        section_end = i_section * 64 + 64
        if i_section % 2 == 0:
            new_trace[(section_start + 128) % 2048:(section_end + 128) % 2048] = trace[section_start:section_end]
        elif i_section > 1:
            new_trace[(section_start - 128) % 2048:(section_end - 128) % 2048] = trace[section_start:section_end]
    new_trace[0:64] = 0
    return new_trace
