import { Test, TestingModule } from '@nestjs/testing';
import { ApointmentService } from './apointment.service';

describe('ApointmentService', () => {
  let service: ApointmentService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ApointmentService],
    }).compile();

    service = module.get<ApointmentService>(ApointmentService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
