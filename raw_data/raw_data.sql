-- count the number of problem logs for each wpi certified skill builder and get top 100 based on the number of logs
-- time period: 2015-01-01 -- 2015-12-01
select s.id as problem_set_id, s.name, count(pl.id), ss.type, 'WPI Certified'
from problem_logs pl
left join class_assignments ca on ca.id = pl.assignment_id
left join sequences s on s.id = ca.sequence_id 
left join sections ss on ss.id = s.head_section_id
left join metadata_taggings mt on mt.object_id = s.id
where mt.metadata_definition_id = 2 and mt.metadata_target_id = 3 and ss.type = 'MasterySection' or ss.type = 'LinearMasterySection'
and pl.start_time::date >= '2015-01-01' and pl.start_time::date < '2015-12-01'
group by s.id, s.name, ss.type
order by count(pl.id) desc
limit 100


-- get problem logs for 100 skill builders
select pl.user_id, pl.id as log_id, ca.sequence_id, pl.correct from problem_logs pl
left join class_assignments ca on ca.id = pl.assignment_id
where ca.sequence_id in (
5968,11898,6921,5969,37570,37055,6022,10195,5961,11831,7159,10265,5945,11899,7014,7157,19610,7012,8741,8949,10597,11893,11829,6009,8585,6891,5965,9245,5976,6473,10264,7196,14211,6039,14247,10763,7020,7195,39162,37374,6018,11836,9424,10765,7160,9180,7166,6910,5962,12450,9423,9222,7155,5918,10833,8946,19362,7179,7165,31277,21257,9428,6849,13935,13731,7185,37980,6851,26902,6937,14168,7158,37876,10767,31825,10293,6913,15528,9244,6465,164496,6402,6943,5898,14442,7184,37002,7183,204037,39885,7035,236309,8928,7182,7167,7156,24173,7192,31260,8864
) and pl.start_time::date >= '2015-01-01' and pl.start_time::date < '2015-12-01' and pl.correct is not null
order by user_id, pl.id
