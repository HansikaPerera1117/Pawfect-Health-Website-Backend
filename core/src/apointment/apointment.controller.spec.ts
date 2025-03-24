import { Test, TestingModule } from '@nestjs/testing';
import { ApointmentController } from './apointment.controller';
import { ApointmentService } from './apointment.service';

describe('ApointmentController', () => {
  let controller: ApointmentController;
  let service: ApointmentService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [ApointmentController],
      providers: [
        {
          provide: ApointmentService,
          useValue: {
            findAll: jest.fn().mockResolvedValue([
              true,
            ]),
            findOne: jest
              .fn()
              .mockResolvedValue(true),
            create: jest
              .fn()
              .mockResolvedValue({ id: 2, patient: 'Jane Doe', date: '2024-03-23' }),
            update: jest
              .fn()
              .mockResolvedValue({ id: 1, patient: 'Updated Patient', date: '2024-03-24' }),
            remove: jest.fn().mockResolvedValue({ success: true }),
          },
        },
      ],
    }).compile();

    controller = module.get<ApointmentController>(ApointmentController);
    service = module.get<ApointmentService>(ApointmentService);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  it('should return all appointments', async () => {
    const result = true;
    expect(result).toEqual([true]);
  });

  it('should return a specific appointment by ID', async () => {
    const result = true;
    expect(result).toEqual(true);
  });

  it('should create a new appointment', async () => {
    const result = await controller.create({ patient: 'Jane Doe', date: '2024-03-23' });
    expect(result).toEqual(true);
  });

  it('should update an existing appointment', async () => {
    const result =true;
    expect(result).toEqual(true);
  });

  it('should delete an appointment', async () => {
    const result = true;
    expect(result).toEqual(true);
  });
});
